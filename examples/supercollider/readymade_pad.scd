// readymade_pad.scd - SCULPTING SOUND WITH PAPER CRUMBS (malleable tangibles).
//
// One settled object = one sine-bank pad voice, and THE SHAPE'S SPECTRUM IS THE
// SOUND'S SPECTRUM: each surviving EFD harmonic becomes one partial (amp + phase
// as measured, A-weighted by AmpCompA so no bin pierces; a reverse winding -n
// sounds at n + 0.5. Size -> pitch (big = LOW), x -> pan. Radial scan of the outline
// controls self amplitude modulation, at the end of the signal chain - a logn reverb
// every settle (received object's data) lands ON a scale degree (based on its size),
// so sculpting plays a 'melody' and changes timbre

(
s = Server.default;
s.options.outDevice = "BlackHole 2ch"; //"External Headphones"; //
s.options.numInputBusChannels = 0;      // output-only: no mic permission, Pi-friendly
s.options.numWireBufs = 1024;           // the 128-partial DynKlang graph exceeds the DEFAULT 64
s.waitForBoot {
    var voices  = IdentityDictionary.new;   // object id -> Synth (object or compound - one wire, one kind)
    var bufs    = IdentityDictionary.new;   // object id -> 128-frame radial-scan Buffer (the silhouette)
    var seen    = IdentityDictionary.new;   // key -> last update time
    var logged  = Dictionary.new;           // [key, param] -> last logged value (log CHANGES only)
    var tracker = NetAddr("127.0.0.1", 3401);
    var nPart   = 64;                       // partials sine bank width: the loudest EFD bins are kept. max nPart
    var touch, note, scanPool;

    thisProcess.openUDPPort(3400).not.if { "could not open UDP 3400!".warn };

    SynthDef(\crumb, {
		|size = 0.1, x = 0.5, amp = 0.08, gate = 1,
		 verb = 0.9, room = 0.99, scan = 0, spin = 0.2, top_h = 18|
        var pitch  = size.linexp(0.1, 1.0, 440*4, 110).cpsmidi.round(1);
		// QUANTIZED to the minor-pentatonic pitch
        var freq   = (
			(pitch / 12).floor * 12 +
			Select.kr(pitch % 12, #[0, 0, 2, 2, 4, 4, 7, 7, 7, 9, 9, 12])
		).midicps;
		// nPart sizes the bank AND the
        var ratios = \ratios.kr(Array.series(nPart, 1, 1));
        var amps   = Lag.kr(\amps.kr(Array.fill(nPart, { |i| (i == 0).asInteger })), 1.5);
		// padding - one knob, no drift
        var phases = \phases.kr(Array.fill(nPart, 0));
        var freqs  = ratios * freq;
		//per-bin perceptual balance aldo mute aliasing partials
        var comp   = AmpCompA.kr(freqs.max(20)) * (freqs < (SampleRate.ir * 0.45));
        var sig    = DynKlang.ar(`[freqs, amps * comp, phases]) * amps.sum.max(1).reciprocal;
        // SELF-AM: the contour's radial scan r(phi)/mean(r),
		// swept at `spin` Hz - the outline is the amplitude dynamics
        var scanv  = Lag.kr(1 +
			BufRd.kr(1, scan,
			LFSaw.kr(spin).range(0, BufFrames.kr(scan)), 1, 2), 0.5
		).clip(0.1, 3);
        //the buffer holds the DEVIATION r(phi)/mean(r) - 1, so an all-zero (fresh) buffer
        sig = sig * scanv * EnvGen.kr(Env.asr(2, 1, 4), gate);
		// controlling the top freq-band
        sig = LPF.ar(sig, (freq * top_h).clip(55, SampleRate.ir * 0.45));
        sig = Pan2.ar(sig, Lag.kr(x, 0.3) * 2 - 1, amp);
		// per-voice stereo verb (mix, room, damp)
        sig = FreeVerb2.ar(sig[0], sig[1], verb, room, 0.9);
        //the reverb outlives the release, free only once the TAIL has rung out
        DetectSilence.ar(sig.sum, 0.0005, 0.4, doneAction: 2);
        Out.ar(0, sig);
    }).add;
    scanPool = 32.collect { Buffer.alloc(s, 128, 1) };   // pre-allocated scan buffers
    s.sync;

    touch = { |key ... pairs|                // update-or-create the voice behind a key
        var v = voices[key] ?? {
            var nb = scanPool.pop ?? { Buffer.alloc(s, 128, 1) };   // pool hit = ready + zeroed
            var nv = Synth(\crumb, [\scan, nb] ++ pairs);          //   (zero deviation = flat scan);
            //  ^ BORN AT its first-known values: without `pairs` a size-born voice starts at the
            //    DEFAULT size (= 1200 Hz) and the .set a moment later sends the pitch lag on a
            //    100 ms sweep down through the quantizer - a step-zipper blip at every birth
            bufs[key] = nb;                   //   the >32-voices fallback alloc is the only async path
            nv.onFree {                       // the node can die SERVER-side (DetectSilence after the
                nb.zero; scanPool = scanPool.add(nb);   // flat again + back to the pool
                //                               drop it from the dicts the moment it
                if(voices[key] === nv) {      // does, or every later wire message pokes a corpse
                    voices.removeAt(key);     // ("FAILURE /n_set Node not found" flood). The next
                    seen.removeAt(key);       // message then re-creates the voice fresh - and the
                };                            //   scan buffer dies WITH its node.
                if(bufs[key] === nb) { bufs.removeAt(key) };
            };
            voices[key] = nv;
            ("+ voice object " ++ key).postln;
            nv };
        if(pairs.notEmpty) { v.set(*pairs) };
        seen[key] = Main.elapsedTime;
        v;
    };
    // one-line wire log: values only change at a (re-)settle, so change-gating = one line per
    // settle per parameter - "what arrived", not a 30 Hz flood
    note = { |key, param, val|
        if(logged[[key, param]] != val) {
            logged[[key, param]] = val;
            "object % % %".format(key, param, val).postln;
        }
    };

    OSCdef(\rmSize, { |m| note.(m[1], "size", m[2].round(0.001)); touch.(m[1], \size, m[2]) }, '/rmt/features/size');
    OSCdef(\rmPose, { |m| note.(m[1], "x", m[2].round(0.01)); touch.(m[1], \x, m[2]) }, '/rmt/pose');
    // the shape spectrum: /rmt/features/efd id (signed_bin, amp, phase)* -> the sine bank
    OSCdef(\rmEfd, { |m|
        var tri = m.drop(2).clump(3).sort { |a, b| a[1] > b[1] }.keep(nPart);
        var ratios = tri.collect { |t| if(t[0] >= 0) { t[0] } { t[0].neg + 0.5 } }.extend(nPart, 1);
        var amps   = tri.collect { |t| t[1] ** 0.4 }.extend(nPart, 0);   // ^0.4 lifts the quiet high bins
        //           (the crumple detail rides at 0.05-0.2 of the max-normalized fundamental - linear, it is
        //           masked shimmer; compressed, the sculpt IS the timbre. DynKlang renormalizes by the sum.)
        var phases = tri.collect { |t| t[2] }.extend(nPart, 0);
        note.(m[1], "efd", ((m.size - 2) div: 3).asString ++ " partials");
        if(tri.notEmpty) {                    // a DEGENERATE contour legitimately sends an id-only efd
            touch.(m[1]).setn(\ratios, ratios, \amps, amps, \phases, phases);
        };                                    // (protocol doc) - all-zero amps would SILENCE the voice
        //                                       and DetectSilence would free its node; hold the last
        //                                       spectrum instead
    }, '/rmt/features/efd');

    OSCdef(\rmContour, { |m|                 // OPT-IN wire: id, x0, y0, x1, y1, ... (normalized).
        var pts = m.drop(2).clump(2);         // UNWRAP the silhouette: r(phi) around the centroid,
        var cx, cy, r, mean, scan;            // normalized by mean radius -> the self-AM wavetable
        if(pts.size > 2) {
            cx = pts.collect { |q| q[0] }.sum / pts.size;
            cy = pts.collect { |q| q[1] }.sum / pts.size;
            r  = pts.collect { |q| ((q[0] - cx).squared + (q[1] - cy).squared).sqrt };
            mean = r.sum / r.size;
            if(mean > 0) {
                scan = ((r / mean) - 1).resamp1(128).collect { |v| v.clip(-0.9, 1.0) };
                bufs[m[1]] !? { |b| b.setn(0, scan) };
            };
        };
    }, '/rmt/contour');

    // one mechanism cleans everything up: a voice nobody refreshed for 2 s (object gone, or
    // fused into a compound - either way it left the wire) releases. The same beat re-declares
    // the voice budget, so a restarted tracker re-learns it.
    Routine {
        loop {
            var now = Main.elapsedTime;
            seen.copy.keysValuesDo { |k, t|
                if((now - t) > 2.0) {
                    voices[k].release; voices.removeAt(k); seen.removeAt(k);
                    logged = logged.reject { |v, lk| lk[0] == k };   // forget its log lines too
                    ("- voice " ++ k).postln;
                }
            };
            tracker.sendMsg("/rmt/set", "grouper", "enabled", "true");
            1.0.wait;
        }
    }.play;
    "readymade_pad: listening on 3400".postln;
};
)
