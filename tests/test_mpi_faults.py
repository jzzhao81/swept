"""Multi-rank error paths (REVIEW_2026-10-17 items 2 and 3, PROPOSAL_v2 section 4b).

Run from the serial suite: each case is a separate `mpiexec -n N` job of
mpi_fault_worker.py, started in its own process group and killed as a group after
TIMEOUT s, so a regression shows up as a failure rather than a hung suite.  Skipped
inside an MPI job, and skipped (with the reason) only when MPI is clearly unavailable
here: no launcher, no mpi4py, or a launcher that cannot start (e.g. sockets refused).
A healthy control that starts but misbehaves is a failure."""
import itertools
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # parent of the swept package
from swept import mpiutils as mpi

WORKER = Path(__file__).resolve().with_name("mpi_fault_worker.py")
TIMEOUT = 60
SIZES = (2, 3)
LAUNCH_FAILURES = ("cannot open socket", "Operation not permitted", "error setting up the bootstrap",
                   "HYD_", "unable to launch")


def mpiexec():
    exe = Path(sys.executable).with_name("mpiexec")
    return str(exe) if exe.exists() else shutil.which("mpiexec")


def _signal_group(pgid, sig):
    try:
        os.killpg(pgid, sig)
    except ProcessLookupError:                    # no process left in the group
        pass


def supervise(cmd, env=None, timeout=TIMEOUT, grace=5):
    """Run cmd in its own process group; returns (returncode or None on timeout, stdout, stderr).

    On timeout the whole group gets SIGTERM and, after at most `grace` s, SIGKILL in any
    case: workers can outlive the launcher (e.g. ignoring SIGTERM) and keep the output
    pipes open, so the launcher having exited proves nothing.  Every read is bounded;
    pipes still held by a process outside the group are abandoned."""
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env,
                            start_new_session=True)
    try:
        out, err = proc.communicate(timeout=timeout)
        return proc.returncode, out, err
    except subprocess.TimeoutExpired:
        pass
    _signal_group(proc.pid, signal.SIGTERM)
    try:
        out, err = proc.communicate(timeout=grace)
    except subprocess.TimeoutExpired:
        out = err = None
    _signal_group(proc.pid, signal.SIGKILL)       # always, also when the launcher is gone
    if out is None:
        try:
            out, err = proc.communicate(timeout=grace)
        except subprocess.TimeoutExpired:
            for pipe in (proc.stdout, proc.stderr):
                pipe.close()
            proc.kill()
            proc.wait(timeout=grace)
            out, err = "", "(output abandoned: pipes held open by a process outside the job's group)"
    return None, out, err


def run_job(n, case, tmp):
    """(returncode or None on timeout, stdout, stderr, markers by rank)."""
    code, out, err = supervise([mpiexec(), "-n", str(n), sys.executable, "-B", "-u", str(WORKER), case],
                               env=dict(os.environ, TMPDIR=tmp))
    marks, decoder, pos = {}, json.JSONDecoder(), out.find("MARK {")
    while pos >= 0:                       # robust to markers that share a line
        m, end = decoder.raw_decode(out, pos + 5)
        marks.setdefault(m["rank"], []).append(m)
        pos = out.find("MARK {", end)
    return code, out, err, marks


@unittest.skipIf(mpi.size > 1, "starts its own mpiexec jobs; run it from the serial suite")
class MpiFaultTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if mpi.comm is None:
            raise unittest.SkipTest("mpi4py not available")
        if mpiexec() is None:
            raise unittest.SkipTest("no mpiexec found")
        cls.tmp = tempfile.mkdtemp()
        code, out, err, marks = run_job(2, "none", cls.tmp)
        if code not in (0, None) and not marks and any(s.lower() in (out + err).lower()
                                                       for s in LAUNCH_FAILURES):
            shutil.rmtree(cls.tmp)
            raise unittest.SkipTest(f"mpiexec cannot start a job here (exit {code}): "
                                    f"{(err or out).strip().splitlines()[0][:200]}")
        cls.control = (code, out, err, marks)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def job(self, n, case):
        code, out, err, marks = run_job(n, case, self.tmp)
        info = f"case {case}, {n} ranks, exit {code}\n--- stdout\n{out[-3000:]}\n--- stderr\n{err[-3000:]}"
        self.assertIsNotNone(code, "timed out (hang)\n" + info)
        return code, marks, info

    def final(self, marks, n, info):
        """The last marker of every rank."""
        self.assertEqual(sorted(marks), list(range(n)), "missing rank markers\n" + info)
        return [marks[r][-1] for r in range(n)]

    def test_healthy_control(self):
        code, out, err, marks = self.control
        info = f"exit {code}\n--- stdout\n{out[-3000:]}\n--- stderr\n{err[-3000:]}"
        self.assertEqual(code, 0, info)
        self.assertEqual([m["outcome"] for m in self.final(marks, 2, info)], ["returned"]*2, info)
        for n in SIZES[1:]:
            code, marks, info = self.job(n, "none")
            self.assertEqual(code, 0, info)
            self.assertEqual([m["outcome"] for m in self.final(marks, n, info)], ["returned"]*n, info)

    def check_collective(self, case, what, failed, error, extra=None):
        for n in SIZES:
            with self.subTest(case=case, n=n):
                code, marks, info = self.job(n, case)
                self.assertEqual(code, 0, info)
                bad_rank = n - 1 if failed == "last" else failed
                for m in self.final(marks, n, info):
                    self.assertEqual(m["outcome"], "CollectiveError", info)
                    self.assertIn(f"{what} failed on 1 rank(s): rank {bad_rank}: {error}", m["msg"], info)
                    for key, value in (extra or {}).items():
                        self.assertEqual(m[key], value, info)

    def test_rank0_failures(self):
        for case, what in [("plane", "plane_eigens"), ("mesh_e", "mesh_eigens (energies)"),
                           ("mesh_v", "mesh_eigens (eigenvectors)"), ("empty_eigvals", "get_bulk_eigvals"),
                           ("eprojs_multi", "get_bulk_eprojs_multi"), ("proj_dos", "get_proj_dos")]:
            self.check_collective(case, what, 0, "LinAlgError")

    def test_nonzero_rank_failure(self):
        self.check_collective("nonzero_rank", "get_bulk_eigvals", "last", "LinAlgError")

    def test_receive_buffer_failure_enters_no_data_collective(self):
        self.check_collective("recv_alloc", "get_bulk_eigvals", "last", "MemoryError",
                              extra={"allgatherv_calls": 0})

    def test_devide_array_preparation_failure(self):
        self.check_collective("devide_prep", "devide_array (preparation)", "last", "MemoryError")

    def test_log_open_failure(self):
        for n in SIZES:
            with self.subTest(n=n):
                code, marks, info = self.job(n, "log_open")
                self.assertEqual(code, 0, info)
                for m in self.final(marks, n, info):
                    self.assertEqual((m["outcome"], m["load_input_calls"]), ("status 1", 0), info)

    def test_nonfinite_reduced_data_returns_1_everywhere(self):
        for n in SIZES:
            with self.subTest(n=n):
                code, marks, info = self.job(n, "nonfinite_main")
                self.assertEqual(code, 0, info)
                for m in self.final(marks, n, info):
                    self.assertEqual(m["outcome"], "status 1", info)
                self.assertIn("error: NonFiniteResult: mesh_eigens: non-finite band energies", info)

    def test_cli_uncaught_error_aborts_the_job(self):
        for n, (case, error) in itertools.product(SIZES, [("cli_abort", "RuntimeError"),
                                                          ("cli_abort_fpe", "FloatingPointError")]):
            with self.subTest(n=n, case=case):
                code, out, err, marks = run_job(n, case, self.tmp)
                info = f"exit {code}\n--- stdout\n{out[-3000:]}\n--- stderr\n{err[-3000:]}"
                self.assertIsNotNone(code, "timed out: the Abort fallback did not end the job\n" + info)
                self.assertNotEqual(code, 0, info)
                self.assertEqual(sorted(marks), list(range(n)), info)      # all ranks had started
                self.assertTrue(all(m[-1]["outcome"] == "started" for m in marks.values()), info)
                self.assertIn(f"{error}: injected uncaught error on rank 0", err + out, info)
                self.assertIn("mpi_abort", (err + out).lower(), info)


LAUNCHER = """
import subprocess, sys, time
child = subprocess.Popen([sys.executable, "-c", "import os, signal, time; "
                          "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                          "print('child', os.getpid(), flush=True); time.sleep(300)"])
print("launcher up", flush=True)
time.sleep(300)
"""


@unittest.skipIf(mpi.size > 1, "process-group test; run it from the serial suite")
class SupervisorTests(unittest.TestCase):
    def test_worker_ignoring_sigterm_after_launcher_exit(self):
        # the launcher dies on SIGTERM, its child ignores it and keeps the pipes open
        start = time.monotonic()
        code, out, err = supervise([sys.executable, "-c", LAUNCHER], timeout=2, grace=2)
        elapsed = time.monotonic() - start
        self.assertIsNone(code)
        self.assertLess(elapsed, 2 + 2*2 + 3, f"supervisor took {elapsed:.1f} s")
        self.assertIn("launcher up", out)
        child = int(out.split("child ")[1].split()[0])
        for _ in range(40):                          # reparented, reaped by init
            try:
                os.kill(child, 0)
            except ProcessLookupError:
                break
            time.sleep(0.05)
        else:
            self.fail(f"child {child} survived the supervisor")

    def test_normal_exit(self):
        self.assertEqual(supervise([sys.executable, "-c", "print('hi')"], timeout=20)[:2], (0, "hi\n"))


if __name__ == "__main__":
    unittest.main()
