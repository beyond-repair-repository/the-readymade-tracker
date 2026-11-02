using System;
using System.Collections.Generic;
using System.Linq;
using System.Net.Sockets;
using System.Text;
using System.Threading.Tasks;

namespace RmtWorld
{
    // --------------------------------------------------------------------------------------------- //
    //  MjpegClient — the tracker's [mjpeg] stream (multipart/x-mixed-replace) -> latest JPEG bytes. //
    //  THREADLESS: gamma hot-reload kills processes with threads parked in user C#, so the mainloop //
    //  drains the socket non-blockingly each frame; the kernel buffer holds between frames.         //
    //  Patch:  MjpegClient(host, port) -> Update -> If NewFrame [ SKImage FromEncodedData ] -> S+H  //
    // --------------------------------------------------------------------------------------------- //

    public class MjpegClient : IDisposable
    {
        readonly string _host; readonly int _port;
        TcpClient _tcp; NetworkStream _net; Task _connecting;
        readonly System.IO.MemoryStream _buf = new();
        readonly byte[] _chunk = new byte[65536];
        byte[] _frame; int _cooldown; string _status = "idle";

        public MjpegClient(string host = "192.168.0.107", int port = 8080)
        { _host = host; _port = Math.Max(1, port); }

        public void Update(out byte[] Jpeg, out bool NewFrame, out string Status)
        {
            NewFrame = false;
            try
            {
                if (_net != null) NewFrame = Drain();
                else if (_connecting != null) FinishConnect();
                else if (--_cooldown <= 0) BeginConnect();
            }
            catch (Exception e) { Drop("error: " + e.Message); }
            Jpeg = _frame; Status = _status;
        }

        void BeginConnect()
        {
            _tcp = new TcpClient();
            _connecting = _tcp.ConnectAsync(_host, _port);
            _status = $"connecting {_host}:{_port}";
        }

        void FinishConnect()
        {
            if (!_connecting.IsCompleted) return;              // poll - never block the mainloop
            if (_connecting.IsFaulted || !_tcp.Connected)
            { var _ = _connecting.Exception; Drop("connect failed"); return; }
            _connecting = null;
            _net = _tcp.GetStream();
            var get = System.Text.Encoding.ASCII.GetBytes($"GET / HTTP/1.1\r\nHost: {_host}\r\n\r\n");
            _net.Write(get, 0, get.Length);
            _status = "streaming";
        }

        bool Drain()
        {
            int budget = 1 << 20;                              // <= 1 MB per mainloop frame
            while (budget > 0 && _net.DataAvailable)
            {
                int n = _net.Read(_chunk, 0, Math.Min(_chunk.Length, budget));
                if (n <= 0) { Drop("closed by server"); return false; }
                _buf.Write(_chunk, 0, n); budget -= n;
            }
            if (_buf.Length > (8 << 20)) _buf.SetLength(0);    // runaway garbage: hard resync
            return Extract();
        }

        /// Latest COMPLETE JPEG in the buffer (SOI FFD8 .. EOI FFD9 - FF-escaping makes both
        /// unambiguous). HTTP + multipart headers between parts fall away in the SOI scan.
        bool Extract()
        {
            var b = _buf.GetBuffer(); int len = (int)_buf.Length;
            int start = -1, lastEnd = -1; byte[] found = null;
            for (int i = 0; i + 1 < len; i++)
            {
                if (b[i] != 0xFF) continue;
                if (b[i + 1] == 0xD8) { if (start < 0) start = i; }
                else if (b[i + 1] == 0xD9 && start >= 0)
                {
                    found = new byte[i + 2 - start];
                    Array.Copy(b, start, found, 0, found.Length);
                    lastEnd = i + 2; start = -1;
                }
            }
            if (found == null)
            {
                if (start > 0) TrimTo(start);                  // keep the partial frame's tail only
                return false;
            }
            TrimTo(lastEnd);                                   // stale frames before it: skipped
            _frame = found; return true;
        }

        void TrimTo(int offset)
        {
            var b = _buf.GetBuffer(); int rest = (int)_buf.Length - offset;
            if (rest > 0) Array.Copy(b, offset, b, 0, rest);
            _buf.SetLength(rest); _buf.Position = rest;
        }

        void Drop(string why)
        {
            _status = why; _cooldown = 90;                     // ~1.5 s before retry
            _connecting = null;
            try { _net?.Dispose(); } catch { }
            try { _tcp?.Close(); } catch { }
            _net = null; _tcp = null;
        }

        public void Dispose() => Drop("disposed");             // gamma disposes on recompile: port freed
    }
}
