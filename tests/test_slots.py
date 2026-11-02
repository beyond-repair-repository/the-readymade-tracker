"""Display slots — a small, RECYCLED label per confirmed object (1,2,3…) shown by outputs instead of
the ever-climbing raw id: bounded, freed on eviction (lowest reused), and carried across a re-id."""
from lib.scene import Detection, Scene


def _mint_confirmed(s, x, y):
    with s.writing("observe"):
        oid = s.mint(Detection((float(x), float(y)), (x, y, 10, 10), 100.0, None))
    with s.writing("classify"):
        s.set(oid, "confirmed", True)
    return oid


def _slots(s):
    return {o.id: o.slot for o in s.snapshot().objects}


def test_slots_are_small_recycled_and_transferred():
    s = Scene()
    s.begin_tick(0, ts=0.0)
    a = _mint_confirmed(s, 10, 10)
    b = _mint_confirmed(s, 20, 20)
    c = _mint_confirmed(s, 30, 30)
    sl = _slots(s)
    assert {sl[a], sl[b], sl[c]} == {1, 2, 3}          # first three confirmed → labels 1,2,3 (not raw ids)
    b_slot = sl[b]
    s.end_tick()

    s.begin_tick(1, ts=1.0)                             # evict b → its slot frees
    with s.writing("identity"):
        s.retire(b)
    s.snapshot()
    s.end_tick()

    s.begin_tick(2, ts=2.0)                             # a fresh confirmed object reuses the LOWEST free slot
    d = _mint_confirmed(s, 40, 40)
    assert _slots(s)[d] == b_slot                       # recycled b's freed label — bounded, not a new high number

    s.begin_tick(3, ts=3.0)                             # a re-id (rewrite) carries the slot with the object
    a_slot = _slots(s)[a]
    with s.writing("identity"):
        s.rewrite_id(a, 999)
    assert _slots(s)[999] == a_slot                     # same label survives the id change


def test_unconfirmed_object_has_no_slot():
    s = Scene()
    s.begin_tick(0, ts=0.0)
    with s.writing("observe"):
        oid = s.mint(Detection((5.0, 5.0), (5, 5, 8, 8), 64.0, None))
    assert _slots(s)[oid] is None                        # a fresh un-confirmed mint burns no label


# --------------------------------------------------------------------------- #
# input-thread fatals: logged + marked, never a silent pseudo-EOF
# --------------------------------------------------------------------------- #

def test_input_thread_marks_and_logs_a_producer_fatal():
    """A backend/preprocess exception used to end the run indistinguishably from EOF (the tracker's
    own log recorded nothing). It must now be logged and carried on ``.error``."""
    import time

    from lib.loop import InputThread, LatestFrameSlot

    logged = []

    class Log:
        def error(self, msg):
            logged.append(msg)

        def __getattr__(self, name):                  # info/warning/… no-ops
            return lambda *a, **k: None

    def bad_producer():
        raise ValueError("decoder exploded")

    it = InputThread(bad_producer, LatestFrameSlot(), logger=Log()).start()
    for _ in range(100):                              # the thread should die almost immediately
        if it.done:
            break
        time.sleep(0.01)
    assert it.done
    assert isinstance(it.error, ValueError)
    assert any("producer failed" in m for m in logged)


def test_input_thread_eof_is_not_an_error():
    import time

    from lib.loop import InputThread, LatestFrameSlot

    def eof_producer():
        raise StopIteration

    it = InputThread(eof_producer, LatestFrameSlot()).start()
    for _ in range(100):
        if it.done:
            break
        time.sleep(0.01)
    assert it.done and it.error is None               # a normal end-of-source stays a normal end


def test_input_thread_exposes_is_alive():
    # serve()'s shutdown loop calls is_alive() on BOTH threads (the pipeline threading.Thread AND the
    # InputThread) to spot a wedged thread before closing resources under it — so InputThread must expose
    # is_alive() like it does start/join/stop, or every shutdown/restart AttributeErrors mid-teardown.
    import time

    from lib.loop import InputThread, LatestFrameSlot

    def eof_producer():
        raise StopIteration

    it = InputThread(eof_producer, LatestFrameSlot()).start()
    assert isinstance(it.is_alive(), bool)            # callable + returns a bool (mirrors threading.Thread)
    for _ in range(100):
        if it.done:
            break
        time.sleep(0.01)
    it.join(timeout=1.0)
    assert it.is_alive() is False                     # stopped after EOF + join — what the shutdown loop checks
