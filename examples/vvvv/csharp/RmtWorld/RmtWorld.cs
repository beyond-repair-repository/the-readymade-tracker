// RmtWorld.cs — TheReadymadeTracker /rmt wire -> everyday objects with per-frame deltas + skins.
//
// RmtWorld OWNS NOTHING: gamma's OSCServer owns the socket; this is a pure state machine.
//   OSCServer (3400) -Data-> ForEach (Reactive) [ OSCMessage Split -> Feed(Address, TypeTags,
//   Arguments) ]   ·   mainloop: Update(FlipY) -> Added / Updated / Removed / All
// Feed parses the RAW big-endian OSC bytes VL.IO.OSC hands over and is safe from ANY thread
// (one lock guards all state). Deltas commit on /rmt/alv; an object is BORN (Added) only at
// its first SETTLED commit, whose bundle carries everything - pose, features, contour.
// Skins: CaptureSkin cuts the object's sprite from the decoded MJPEG frame at Added time
// (bbox crop, contour-masked, transparent outside) and stores it ON the object.

using System;
using System.Collections.Generic;
using SkiaSharp;
using Stride.Core.Mathematics;
using VL.Lib.Collections;

namespace Main;

/// The tracker's settle lifecycle, as streamed on /rmt/obj (arg 2, uppercase strings).
public enum RmtState
{
    UNKNOWN,    // client sentinel: no /rmt/obj seen yet, or a name TryParse didn't know
    SETTLING,   // birth debounce — ON THE WIRE with state only (no pose/features/contour
                //   until capture); this client withholds Added until the first settle
    RESTING,    // settled and committed — features, pose and contour are current
    MOVED,      // was resting, now displaced — pose HELD at the last settle while it travels
    HIDDEN,     // covered by a hand or lifted — the husk persists until the tracker calls it gone
}

public class MyEverydayObject
{
    public int Id;
    public RmtState State = RmtState.UNKNOWN;
    public Vector2 Position;                       // scene space: -1..1, aspect-true
    public float Angle;                            // unwrapped accumulated radians (wire theta)
    public float Confidence;
    public Spread<Vector2> Shape = Spread<Vector2>.Empty;   // /rmt/contour, ABSOLUTE scene space
                                                            //   (opt-in: --set oscout send "all, contour")
    public SKImage Skin;                    // the captured sprite: an INDEPENDENT immutable copy
    public Vector2 SkinOffset;              // sprite-center MINUS o.Position at capture (scene units)
    public Vector2 SkinSize;                // the sprite's size in SCENE units at capture

    // the person-facing feature set, all 0..1 unless noted (see docs/OSC_OUT_PROTOCOL.txt)
    public float Circularity;
    public float Rectangularity;
    public float Solidity;
    public float Eccentricity;
    public float Size;
    public float LobeCount;
    public float Spikiness;
    public float ReflectionalSymmetry;
    public Spread<Vector3> Efd = Spread<Vector3>.Empty;   // (signed bin, amp, phase) triples

    public bool IsNew;                             // true during the ONE Update() after it appeared
    public bool IsGone;                            // set forever ON THIS INSTANCE at retirement; an id
                                                   //   is never reused for a DIFFERENT object, but the
                                                   //   SAME object can return (gone-pool, ~5 s) as a
                                                   //   FRESH Added instance
    internal bool Announced;
    internal bool Dirty;

    public Color4 Color => StateColor(State);

    public static Color4 StateColor(RmtState s) => s switch
    {
        RmtState.RESTING => new Color4(0x00 / 255f, 0xdc / 255f, 0x00 / 255f, 1f),
        RmtState.MOVED => new Color4(0xff / 255f, 0x8c / 255f, 0x00 / 255f, 1f),
        RmtState.SETTLING => new Color4(0x00 / 255f, 0xdc / 255f, 0xdc / 255f, 1f),
        RmtState.HIDDEN => new Color4(0xa0 / 255f, 0xa0 / 255f, 0xa0 / 255f, 1f),
        _ => new Color4(0xc8 / 255f, 0xc8 / 255f, 0xc8 / 255f, 1f),
    };

    public void SetSkin(SKImage skin, Vector2 offset, Vector2 size)
    { Skin = skin; SkinOffset = offset; SkinSize = size; }

    /// Physics-ready Shape, RELATIVE to Position: decimated + forced CCW. Feed Aether from
    /// CollisionHull below; draw the raw Shape. (Delete both if the patch no longer uses them.)
    public Spread<Vector2> PhysicsShape(float minEdge = 0.02f)
    {
        var shape = Shape; var pos = Position;        // SNAPSHOT: Feed may swap these mid-loop
        int n = shape.Count;
        if (n < 3) return Spread<Vector2>.Empty;
        var pts = new List<Vector2>();
        Vector2 last = default;
        for (int k = 0; k < n; k++)
        {
            var p = shape[k] - pos;
            if (k == 0 || (p - last).LengthSquared() >= minEdge * minEdge) { pts.Add(p); last = p; }
        }
        while (pts.Count >= 3 && (pts[pts.Count - 1] - pts[0]).LengthSquared() < minEdge * minEdge)
            pts.RemoveAt(pts.Count - 1);
        if (pts.Count < 3) return Spread<Vector2>.Empty;
        float area2 = 0;
        for (int k = 0; k < pts.Count; k++)
        { var p = pts[k]; var q = pts[(k + 1) % pts.Count]; area2 += p.X * q.Y - q.X * p.Y; }
        if (MathF.Abs(area2) < minEdge * minEdge) return Spread<Vector2>.Empty;
        if (area2 < 0) pts.Reverse();
        var sb = new SpreadBuilder<Vector2>();
        foreach (var p in pts) sb.Add(p);
        return sb.ToSpread();
    }

    /// Collision proxy for Aether: the CONVEX HULL of PhysicsShape, capped at maxVerts (<= 8).
    public Spread<Vector2> CollisionHull(int maxVerts = 8, float minEdge = 0.02f)
    {
        var poly = PhysicsShape(minEdge);
        if (poly.Count < 3) return poly;
        static float Cross(Vector2 o, Vector2 a, Vector2 b)
            => (a.X - o.X) * (b.Y - o.Y) - (a.Y - o.Y) * (b.X - o.X);
        var pts = new List<Vector2>(poly);
        pts.Sort((u, v) => u.X != v.X ? u.X.CompareTo(v.X) : u.Y.CompareTo(v.Y));
        var hull = new List<Vector2>();
        for (int pass = 0; pass < 2; pass++)
        {
            int start = hull.Count;
            for (int k = 0; k < pts.Count; k++)
            {
                var p = pts[pass == 0 ? k : pts.Count - 1 - k];
                while (hull.Count - start >= 2 && Cross(hull[hull.Count - 2], hull[hull.Count - 1], p) <= 0)
                    hull.RemoveAt(hull.Count - 1);
                hull.Add(p);
            }
            hull.RemoveAt(hull.Count - 1);
        }
        if (hull.Count > maxVerts)
        {
            var kept = new List<Vector2>();
            for (int j = 0; j < maxVerts; j++)
            {
                var dir = new Vector2(MathF.Cos(j * 2f * MathF.PI / maxVerts),
                                      MathF.Sin(j * 2f * MathF.PI / maxVerts));
                int best = 0; float bd = float.MinValue;
                for (int k = 0; k < hull.Count; k++)
                { float d = hull[k].X * dir.X + hull[k].Y * dir.Y; if (d > bd) { bd = d; best = k; } }
                if (!kept.Contains(hull[best])) kept.Add(hull[best]);
            }
            hull = kept;
        }
        if (hull.Count < 3) return Spread<Vector2>.Empty;
        var sb = new SpreadBuilder<Vector2>();
        foreach (var p in hull) sb.Add(p);
        return sb.ToSpread();
    }
}

public class RmtWorld
{
    readonly object _gate = new();                 // Feed: any thread; Update: mainloop. ONE lock.
    readonly Dictionary<int, MyEverydayObject> _objs = new();
    int _w = 1, _h = 1;
    int _frame; int _nObjects;
    bool _committed;
    bool _flipY = true;                            // latch before the first frame and LEAVE it

    readonly List<MyEverydayObject> _added = new(), _updated = new(), _removed = new();
    readonly List<MyEverydayObject> _freshLast = new();   // Added LAST Update: IsNew clears one late

    public void Feed(string address, string typeTags, IReadOnlyList<byte> args)
    {
        if (address == null || typeTags == null || args == null || !address.StartsWith("/rmt")) return;
        var tags = typeTags.Length > 0 && typeTags[0] == ',' ? typeTags.Substring(1) : typeTags;
        var a = new List<object>(tags.Length); int i = 0;
        foreach (var t in tags)
            switch (t)
            {
                case 'i': if (i + 4 > args.Count) return; a.Add(ReadI32(args, i)); i += 4; break;
                case 'f':
                    if (i + 4 > args.Count) return;
                    a.Add(BitConverter.Int32BitsToSingle(ReadI32(args, i))); i += 4; break;
                case 's': a.Add(ReadStr(args, ref i)); break;
                case 'd': case 't': case 'h': i += 8; a.Add(0f); break;
                case 'T': case 'F': case 'N': case 'I': a.Add(0f); break;
                default: return;
            }
        lock (_gate) Handle(address, a, _flipY);
    }

    public void Update(bool flipY,
        out Spread<MyEverydayObject> Added, out Spread<MyEverydayObject> Updated,
        out Spread<MyEverydayObject> Removed, out Spread<MyEverydayObject> All,
        out int FrameIndex, out int ObjectCount, out int Width, out int Height,
        out bool OnCommit)
    {
        lock (_gate)
        {
            _flipY = flipY;
            foreach (var o in _freshLast) o.IsNew = false;   // newborns stayed IsNew for ONE Update
            _freshLast.Clear();
            Added = ToSpread(_added); Updated = ToSpread(_updated); Removed = ToSpread(_removed);
            var all = new SpreadBuilder<MyEverydayObject>();
            foreach (var o in _objs.Values) all.Add(o);
            All = all.ToSpread();
            FrameIndex = _frame; ObjectCount = _nObjects; Width = _w; Height = _h;
            OnCommit = _committed;
            _freshLast.AddRange(_added);
            _added.Clear(); _updated.Clear(); _removed.Clear(); _committed = false;
        }
    }

    void Handle(string addr, IReadOnlyList<object> a, bool flipY)
    {
        switch (addr)
        {
            case "/rmt/frm" when a.Count >= 2:
                _frame = I(a[0]); _nObjects = I(a[1]);
                if (a.Count >= 4) { _w = Math.Max(1, I(a[2])); _h = Math.Max(1, I(a[3])); }
                break;
            case "/rmt/obj" when a.Count >= 3:
                {
                    var o = Touch(I(a[0]));
                    var s = Enum.TryParse<RmtState>(S(a[1]), true, out var st) ? st : RmtState.UNKNOWN;
                    if (o.State != s) { o.State = s; o.Dirty = true; }
                    o.Confidence = F(a[2]); break;
                }
            case "/rmt/pose" when a.Count >= 4:
                {
                    var o = Touch(I(a[0]));
                    var p = Scene(F(a[1]), F(a[2]), flipY); var th = F(a[3]);
                    if (p != o.Position || th != o.Angle) { o.Position = p; o.Angle = th; o.Dirty = true; }
                    break;
                }
            case "/rmt/contour" when a.Count >= 3:
                {
                    var o = Touch(I(a[0]));
                    var sb = new SpreadBuilder<Vector2>();
                    for (int k = 1; k + 1 < a.Count; k += 2) sb.Add(Scene(F(a[k]), F(a[k + 1]), flipY));
                    var shape = sb.ToSpread();
                    if (shape.Count != o.Shape.Count) o.Dirty = true;
                    o.Shape = shape; break;
                }
            case "/rmt/alv": Commit(a); break;
            default:
                if (addr.StartsWith("/rmt/features/") && a.Count >= 1)
                    Feature(addr.Substring("/rmt/features/".Length), a);
                break;
        }
    }

    void Feature(string name, IReadOnlyList<object> a)
    {
        var o = Touch(I(a[0]));
        if (name == "efd")                          // may legitimately arrive ID-ONLY: a degenerate
        {                                           //   contour CLEARS the spectrum
            var sb = new SpreadBuilder<Vector3>();
            for (int k = 1; k + 3 <= a.Count; k += 3)
                sb.Add(new Vector3(F(a[k]), F(a[k + 1]), F(a[k + 2])));
            var efd = sb.ToSpread();
            if (efd.Count != o.Efd.Count) o.Dirty = true;
            o.Efd = efd;
            return;
        }
        if (a.Count < 2) return;
        var v = F(a[1]);
        switch (name)
        {
            case "circularity": if (v != o.Circularity) { o.Circularity = v; o.Dirty = true; } break;
            case "rectangularity": if (v != o.Rectangularity) { o.Rectangularity = v; o.Dirty = true; } break;
            case "solidity": if (v != o.Solidity) { o.Solidity = v; o.Dirty = true; } break;
            case "eccentricity": if (v != o.Eccentricity) { o.Eccentricity = v; o.Dirty = true; } break;
            case "size": if (v != o.Size) { o.Size = v; o.Dirty = true; } break;
            case "lobe_count": if (v != o.LobeCount) { o.LobeCount = v; o.Dirty = true; } break;
            case "spikiness": if (v != o.Spikiness) { o.Spikiness = v; o.Dirty = true; } break;
            case "reflectional_symmetry": if (v != o.ReflectionalSymmetry) { o.ReflectionalSymmetry = v; o.Dirty = true; } break;
        }
    }

    void Commit(IReadOnlyList<object> alive)
    {
        _committed = true;
        var keep = new HashSet<int>(alive.Count);
        foreach (var x in alive) keep.Add(I(x));
        var dead = new List<int>();
        foreach (var id in _objs.Keys) if (!keep.Contains(id)) dead.Add(id);
        foreach (var id in dead)
        {
            var o = _objs[id]; _objs.Remove(id); o.IsGone = true;
            if (_added.Remove(o)) _updated.Remove(o);   // born + gone between two Updates: it was
            else if (o.Announced)                       //   never emitted, so it never existed
            { _updated.Remove(o); _removed.Add(o); }
        }
        foreach (var o in _objs.Values)
        {
            if (!o.Announced)
            {
                if (o.State == RmtState.SETTLING)       // on the wire, but NOT BORN YET — birth =
                { o.Dirty = false; continue; }          //   the first SETTLED commit
                o.Announced = true; o.IsNew = true; _added.Add(o);
            }
            else if (o.Dirty && !_added.Contains(o) && !_updated.Contains(o))
                _updated.Add(o);
            o.Dirty = false;
        }
    }

    MyEverydayObject Touch(int id)
    {
        if (!_objs.TryGetValue(id, out var o)) _objs[id] = o = new MyEverydayObject { Id = id };
        return o;
    }

    Vector2 Scene(float xn, float yn, bool flipY)
    {
        float m = Math.Max(_w, _h);
        return new Vector2((xn - 0.5f) * 2f * ((float)_w / m),
                           (flipY ? 0.5f - yn : yn - 0.5f) * 2f * ((float)_h / m));
    }

    /// Capture o's sprite from the CURRENT decoded frame: bbox crop, contour-masked,
    /// transparent outside. Call from the patch's Added ForEach: CaptureSkin(o, frameImage).
    public void CaptureSkin(MyEverydayObject o, SKImage frame)
    {
        var shape = o.Shape; var pos = o.Position;
        if (frame == null || shape.Count < 3) return;
        float m = Math.Max(_w, _h);
        SKPoint Px(Vector2 p)
        {
            float xn = p.X * m / (2f * _w) + 0.5f;
            float yn = _flipY ? 0.5f - p.Y * m / (2f * _h) : p.Y * m / (2f * _h) + 0.5f;
            return new SKPoint(xn * frame.Width, yn * frame.Height);
        }
        var path = new SKPath();
        path.MoveTo(Px(shape[0]));
        for (int i = 1; i < shape.Count; i++) path.LineTo(Px(shape[i]));
        path.Close();
        var b = path.Bounds; b.Inflate(2, 2);
        var ib = SKRectI.Round(b);
        ib.Intersect(new SKRectI(0, 0, frame.Width, frame.Height));
        if (ib.Width < 2 || ib.Height < 2) return;
        using var surf = SKSurface.Create(new SKImageInfo(ib.Width, ib.Height,
                                          SKColorType.Rgba8888, SKAlphaType.Premul));
        var cv = surf.Canvas;
        cv.Clear(SKColors.Transparent);
        cv.Translate(-ib.Left, -ib.Top);
        cv.ClipPath(path, antialias: true);
        cv.DrawImage(frame, 0, 0);
        float xs = (ib.MidX / (float)frame.Width - 0.5f) * 2f * (_w / m);
        float ys = _flipY ? (0.5f - ib.MidY / (float)frame.Height) * 2f * (_h / m)
                          : (ib.MidY / (float)frame.Height - 0.5f) * 2f * (_h / m);
        float unit = 2f * _w / (m * frame.Width);
        o.SetSkin(surf.Snapshot(),
                  new Vector2(xs, ys) - pos,
                  new Vector2(ib.Width * unit, ib.Height * unit));
    }

    /// Patch-side join for the draw pass: body Tag -> id -> the live object.
    public void TryGet(int id, out MyEverydayObject Obj, out bool Found)
    {
        lock (_gate) Found = _objs.TryGetValue(id, out Obj);
    }

    static int ReadI32(IReadOnlyList<byte> b, int i)
        => (b[i] << 24) | (b[i + 1] << 16) | (b[i + 2] << 8) | b[i + 3];

    static string ReadStr(IReadOnlyList<byte> b, ref int i)
    {
        var sb = new System.Text.StringBuilder();
        while (i < b.Count && b[i] != 0) sb.Append((char)b[i++]);
        i = (i + 4) & ~3; return sb.ToString();
    }

    // the decode boxes exact types — Handle still COERCES, never casts: a tag/arity surprise
    // must degrade to 0, not throw inside the reactive region (that kills the stream silently).
    static int I(object x) => x switch { int i => i, long l => (int)l, float f => (int)f, double d => (int)d, _ => 0 };
    static float F(object x) => x switch { float f => f, double d => (float)d, int i => i, long l => l, _ => 0f };
    static string S(object x) => x as string ?? "";

    static Spread<MyEverydayObject> ToSpread(List<MyEverydayObject> l)
    { var sb = new SpreadBuilder<MyEverydayObject>(); foreach (var o in l) sb.Add(o); return sb.ToSpread(); }
}