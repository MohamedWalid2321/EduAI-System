"""
lockdown.py — Flask blueprint for exam environment security checks.

Provides POST /check-environment which runs a suite of best-effort
detections (virtual machine, remote desktop, screen-capture software)
and returns a structured EnvironmentCheckResult.

All sub-checks are treated as non-detections on any exception, subprocess
timeout, or missing platform support, consistent with the silent-failure
requirement (FR-017).

Windows-only feature — checks that depend on winreg or PowerShell are
skipped on non-Windows platforms and return False.
"""

import platform
import subprocess
import sys

import psutil
from flask import Blueprint, jsonify

# ---------------------------------------------------------------------------
# Blueprint
# ---------------------------------------------------------------------------

lockdown_bp = Blueprint("lockdown", __name__)

# ---------------------------------------------------------------------------
# VM vendor keywords (case-insensitive match against model/processor strings)
# ---------------------------------------------------------------------------

_VM_KEYWORDS = [
    "virtualbox",
    "vmware",
    "qemu",
    "hyper-v",
    "hyperv",
    "xenhvm",
    "parallels",
    "kvm",
]

# Virtual Machine NIC OUI prefixes (lowercase colon-separated, first 3 octets)
_VM_OUI_PREFIXES = [
    "08:00:27",  # VirtualBox
    "00:0c:29",  # VMware ESX
    "00:50:56",  # VMware Workstation
    "00:1c:42",  # Parallels
    "52:54:00",  # QEMU/KVM
    "00:03:ff",  # Hyper-V
]

# ---------------------------------------------------------------------------
# Remote desktop process list (case-insensitive match)
# ---------------------------------------------------------------------------

_RDP_PROCESSES = [
    "teamviewer.exe",
    "anydesk.exe",
    "vncviewer.exe",
    "vncserver.exe",
    "tvnserver.exe",
    "logmein.exe",
    "parsec.exe",
    "rustdesk.exe",
]

# ---------------------------------------------------------------------------
# Screen-capture process list (case-insensitive match)
# ---------------------------------------------------------------------------

_CAPTURE_PROCESSES = [
    "snippingtool.exe",
    "screenclippinghost.exe",
    "sharex.exe",
    "greenshot.exe",
    "obs64.exe",
    "obs32.exe",
    "camtasia.exe",
]

# ---------------------------------------------------------------------------
# Exam-system process allowlist — never flagged as screen capture
# ---------------------------------------------------------------------------
# The violation clip recorder uses ffmpeg (via imageio_ffmpeg) to encode
# WebM clips before uploading them to Bunny CDN.  Those ffmpeg subprocesses
# are part of the exam integrity system and must not be blocked even if a
# future version of _CAPTURE_PROCESSES accidentally includes them.

def _build_capture_allowlist() -> set:
    """Return the set of lower-case process basenames owned by the exam system."""
    allowed: set = set()
    # imageio_ffmpeg embeds its own ffmpeg binary with a versioned filename
    # (e.g. "ffmpeg-win-x86_64-v7.1.exe").  Resolve it at import time so the
    # check is always exact — no glob matching, no partial string comparisons.
    try:
        import imageio_ffmpeg as _imageio_ffmpeg  # type: ignore
        import os as _os
        allowed.add(_os.path.basename(_imageio_ffmpeg.get_ffmpeg_exe()).lower())
    except Exception:
        pass
    # Generic fallback used when imageio_ffmpeg is unavailable (FR-017).
    allowed.add("ffmpeg.exe")
    return allowed

_CAPTURE_PROCESS_ALLOWLIST: set = _build_capture_allowlist()

# ---------------------------------------------------------------------------
# Internal helpers — VM detection
# ---------------------------------------------------------------------------


def _run_powershell(command: str) -> str:
    """
    Run a PowerShell command and return its stripped stdout output.
    Returns an empty string on any error or timeout.
    Raises nothing — all exceptions are swallowed.
    """
    try:
        proc = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", command],
            capture_output=True,
            text=True,
            timeout=3,
        )
        return proc.stdout.strip()
    except subprocess.TimeoutExpired as exc:
        # Windows: TimeoutExpired does not auto-kill — must call kill() explicitly
        try:
            exc.process.kill()
        except Exception:
            pass
        return ""
    except Exception:
        return ""


def _check_vm_wmi() -> tuple[bool, str | None]:
    """Check Win32_ComputerSystem and Win32_DiskDrive model strings for VM keywords."""
    if sys.platform != "win32":
        return False, None

    cs_model = _run_powershell(
        "Get-CimInstance Win32_ComputerSystem | Select-Object -ExpandProperty Model"
    )
    dd_model = _run_powershell(
        "Get-CimInstance Win32_DiskDrive | Select-Object -ExpandProperty Model"
    )

    combined = f"{cs_model} {dd_model}".lower()
    for keyword in _VM_KEYWORDS:
        if keyword in combined:
            return True, f"VM keyword '{keyword}' detected in system model"
    return False, None


def _check_vm_registry() -> tuple[bool, str | None]:
    """Check Windows registry SCSI/Disk enumeration keys for VM vendor strings."""
    if sys.platform != "win32":
        return False, None

    try:
        import winreg  # stdlib, Windows-only

        registry_paths = [
            (winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DEVICEMAP\Scsi"),
            (winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Services\Disk\Enum"),
        ]

        for hive, subkey in registry_paths:
            try:
                key = winreg.OpenKey(hive, subkey)
                index = 0
                while True:
                    try:
                        name, data, _ = winreg.EnumValue(key, index)
                        if isinstance(data, str):
                            data_lower = data.lower()
                            for keyword in _VM_KEYWORDS:
                                if keyword in data_lower:
                                    winreg.CloseKey(key)
                                    return True, f"VM keyword '{keyword}' in registry key {subkey}\\{name}"
                        index += 1
                    except OSError:
                        break
                winreg.CloseKey(key)
            except OSError:
                continue
    except Exception:
        pass

    return False, None


def _check_vm_mac() -> tuple[bool, str | None]:
    """Check NIC MAC address OUI prefixes for known VM vendors.

    Host-only bridge adapters created by a hypervisor installed on a physical
    machine (e.g. VMware Workstation's VMnet1/VMnet8) share the same OUI
    prefixes as guest adapters but do NOT indicate the OS is running inside a
    VM.  We skip any interface whose name contains a host-bridge marker
    ("vmnet", "virtualbox host", "vboxnet", "loopback") to avoid false
    positives on developer machines.
    """
    _HOST_BRIDGE_MARKERS = ("vmnet", "virtualbox host", "vboxnet", "loopback", "vethernet")
    try:
        addrs = psutil.net_if_addrs()
        for iface_name, iface_addrs in addrs.items():
            # Skip known host-only / bridge adapters that are not guest NICs
            if any(marker in iface_name.lower() for marker in _HOST_BRIDGE_MARKERS):
                continue
            for addr in iface_addrs:
                # AF_LINK == 17 on Linux, psutil.AF_LINK on Windows
                if addr.family == psutil.AF_LINK and addr.address:
                    mac = addr.address.lower().replace("-", ":")
                    oui = ":".join(mac.split(":")[:3])
                    if oui in _VM_OUI_PREFIXES:
                        return True, f"VM NIC OUI {oui} detected"
    except Exception:
        pass
    return False, None


def _check_vm_processor() -> tuple[bool, str | None]:
    """Check the CPU brand string for VM keywords."""
    try:
        cpu = platform.processor().lower()
        for keyword in _VM_KEYWORDS:
            if keyword in cpu:
                return True, f"VM keyword '{keyword}' in CPU brand string"
    except Exception:
        pass
    return False, None


def _detect_vm() -> tuple[bool, str | None]:
    """
    Run all VM detection sub-checks.
    Returns (True, reason) on first positive match; (False, None) otherwise.
    """
    for check in (_check_vm_wmi, _check_vm_registry, _check_vm_mac, _check_vm_processor):
        detected, reason = check()
        if detected:
            return True, reason
    return False, None


# ---------------------------------------------------------------------------
# Internal helpers — RDP / remote-desktop detection
# ---------------------------------------------------------------------------


def _detect_rdp() -> tuple[bool, str | None]:
    """
    Check for running remote desktop processes.
    Returns (True, reason) on first positive match; (False, None) otherwise.
    """
    # Process-based detection only — a passive port 3389 LISTEN state is not
    # treated as a violation because Windows 10/11 Pro enables the RDP service
    # by default even when no remote session is ever active.
    try:
        for proc in psutil.process_iter(["name"]):
            try:
                name = (proc.info.get("name") or "").lower()
                if name in _RDP_PROCESSES:
                    return True, f"{proc.info['name']} running"
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    except Exception:
        pass

    return False, None


# ---------------------------------------------------------------------------
# Internal helpers — screen-capture detection
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Screen-capture baseline — populated on the first /check-environment call.
# Processes already running before the exam started are not flagged.
# ---------------------------------------------------------------------------

_capture_baseline: set | None = None


def _detect_screen_capture() -> tuple[bool, str | None]:
    """
    Check for screen-capture application processes that launched AFTER the
    exam started.  On the first call the current process list is saved as a
    baseline; subsequent calls only alert on NEW processes not in that baseline.
    Returns (True, reason) on first positive match; (False, None) otherwise.
    """
    global _capture_baseline

    # Build the set of names currently running in the monitored list
    current_capture_procs: dict[str, str] = {}  # lower_name -> original_name
    try:
        for proc in psutil.process_iter(["name"]):
            try:
                name = (proc.info.get("name") or "").lower()
                if name in _CAPTURE_PROCESSES and name not in _CAPTURE_PROCESS_ALLOWLIST:
                    current_capture_procs[name] = proc.info["name"]
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    except Exception:
        pass

    # First call — record everything currently running as the baseline
    if _capture_baseline is None:
        _capture_baseline = set(current_capture_procs.keys())
        return False, None

    # Subsequent calls — only flag processes that are NEW since the baseline
    for lower_name, original_name in current_capture_procs.items():
        if lower_name not in _capture_baseline:
            return True, f"{original_name} launched during exam"

    return False, None


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------


@lockdown_bp.route("/check-environment", methods=["POST"])
def check_environment():
    """
    POST /check-environment

    Runs VM, RDP, and screen-capture detection checks.
    All checks are best-effort — any exception returns non-detection for that check.

    Returns 200 with EnvironmentCheckResult JSON, or 500 on unexpected error.
    """
    try:
        vm_detected, vm_reason = _detect_vm()
        rdp_detected, rdp_reason = _detect_rdp()
        capture_detected, capture_reason = _detect_screen_capture()

        return jsonify(
            {
                "vm_detected": vm_detected,
                "vm_reason": vm_reason,
                "rdp_detected": rdp_detected,
                "rdp_reason": rdp_reason,
                "screen_capture_detected": capture_detected,
                "screen_capture_reason": capture_reason,
            }
        ), 200

    except Exception as exc:
        return jsonify(
            {
                "code": "CHECK_ERROR",
                "message": f"Environment check failed: {exc}",
            }
        ), 500
