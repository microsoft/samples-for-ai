import platform
import argparse
import logging
import sys
import subprocess
import os
import shutil
import re
import ctypes
import stat
import importlib
import _thread

TOOLSFORAI_OS_WIN = "win"
TOOLSFORAI_OS_LINUX = "linux"
TOOLSFORAI_OS_MACOS = "mac"
sys_info = {
    "OS": None,
    "python": None,
    "GPU": False,
    "CUDA": None,
    "cudnn": None,
    "mpi": None,
    "tensorflow": None,
    "cuda80": False,
    "git": False
}
fail_install = []

if platform.system() == "Windows":
    import winreg
    from ctypes.wintypes import HANDLE, BOOL, DWORD, HWND, HINSTANCE, HKEY

    class ShellExecuteInfo(ctypes.Structure):
        _fields_ = [('cbSize', DWORD),
                    ('fMask', ctypes.c_ulong),
                    ('hwnd', HWND),
                    ('lpVerb', ctypes.c_char_p),
                    ('lpFile', ctypes.c_char_p),
                    ('lpParameters', ctypes.c_char_p),
                    ('lpDirectory', ctypes.c_char_p),
                    ('nShow', ctypes.c_int),
                    ('hInstApp', HINSTANCE),
                    ('lpIDList', ctypes.c_void_p),
                    ('lpClass', ctypes.c_char_p),
                    ('hKeyClass', HKEY),
                    ('dwHotKey', DWORD),
                    ('hIcon', HANDLE),
                    ('hProcess', HANDLE)]

        def __init__(self, **kw):
            ctypes.Structure.__init__(self)
            self.cbSize = ctypes.sizeof(self)
            for name, value in kw.items():
                setattr(self, name, value)

def _init_logger(log_level=logging.INFO):
    logger = logging.getLogger('Microsoft Visual Studio Tools for AI')
    logger.setLevel(log_level)
    logger.propagate = False
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(fmt='%(asctime)s [%(levelname)s] '
                                      '[%(name)s] %(message)s',
                                  datefmt='%H:%M:%S')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def module_exists(module_name):
    try:
        from pkgutil import iter_modules
        return module_name in (name for loader, name, ispkg in iter_modules())
    except:
        return False


def _registry_read(hkey, keypath, value_name):
    try:
        registry_key = winreg.OpenKey(hkey, keypath)
        value, _ = winreg.QueryValueEx(registry_key, value_name)
        winreg.CloseKey(registry_key)
        return value
    except Exception as e:
        logger.debug("Fail to read registry key: {0}, value: {1}, unexpected error: {2}".format(keypath, value_name, e))
        return None


def _registry_write(hkey, keypath, name, value):
    try:
        registry_key = winreg.CreateKeyEx(hkey, keypath)
        winreg.SetValueEx(registry_key, name, 0, winreg.REG_SZ, value)
        winreg.CloseKey(registry_key)
        return True
    except Exception as e:
        logger.debug("Fail to write registry key: {0}, name: {1}, value: {2}, unexpected error: {3}".format(keypath, name, value, e))
        return False

def _registry_delete(hkey, keypath, name):
    try:
        registry_key = winreg.OpenKey(hkey, keypath, 0, winreg.KEY_SET_VALUE)
        winreg.DeleteValue(registry_key, name)
        winreg.CloseKey(registry_key)
    except Exception as e:
        # logger.debug("Fail to delete registry key: {0}, name: {1},  unexpected error: {2}".format(keypath, name, e))
        raise e

def _registry_subkeys(hkey, keypath):
    key = winreg.OpenKey(hkey, keypath, 0, winreg.KEY_READ)
    i = 0
    while True:
        try:
            subkey = winreg.EnumKey(key, i)
            yield subkey
            i += 1
        except WindowsError as e:
            break


def _run_cmd(cmd, args=[], return_stdout=False):
    try:
        p = subprocess.run([cmd, *args], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        stdout = p.stdout.strip()
        stderr = p.stderr.strip()
        status = p.returncode == 0
        logger.debug("========== {:^30} ==========".format("%s : stdout" % cmd))
        for line in filter(lambda x: x.strip(), p.stdout.split('\n')):
            logger.debug(line)
        logger.debug("========== {:^30} ==========".format("%s : stdout end" % cmd))
        logger.debug("========== {:^30} ==========".format("%s : stderr" % cmd))
        for line in filter(lambda x: x.strip(), p.stderr.split('\n')):
            logger.debug(line)
        logger.debug("========== {:^30} ==========".format("%s : stderr end" % cmd))
    except Exception as e:
        logger.debug("Fail to execute command: {0}, unexpected error: {1}".format(cmd, e))
        status = False
        stdout = ""
    if return_stdout:
        return status, stdout
    else:
        return status


def _wait_process(processHandle, timeout=-1):
    try:
        ret = ctypes.windll.kernel32.WaitForSingleObject(processHandle, timeout)
        logger.debug("Wait process return value: %d" % ret)
    except Exception as e:
        logger.debug("Fail to wait process, unexpected error: {0}".format(e))
    finally:
        ctypes.windll.kernel32.CloseHandle(processHandle)


def _run_cmd_admin(cmd, param, wait=True):
    try:
        executeInfo = ShellExecuteInfo(fMask=0x00000040, hwnd=None, lpVerb='runas'.encode('utf-8'),
                                       lpFile=cmd.encode('utf-8'), lpParameters=param.encode('utf-8'),
                                       lpDirectory=None,
                                       nShow=5)
        if not ctypes.windll.shell32.ShellExecuteEx(ctypes.byref(executeInfo)):
            raise ctypes.WinError()
        if wait:
            _wait_process(executeInfo.hProcess)
    except Exception as e:
        # logger.error("Fail to run command {0} as admin, unexpected error: {1}".format(cmd, e))
        logger.error("Fail to run command {0} as admin, unexpected error! Please try to run installer script again!".format(cmd))


def _download_file(url, local_path):
    logger.info("Downloading {0} ...".format(url))
    try:
        import urllib.request
        import ssl
        myssl = ssl.create_default_context()
        myssl.check_hostname = False
        myssl.verify_mode = ssl.CERT_NONE
        with urllib.request.urlopen(url, context=myssl) as fin, \
                open(local_path, 'ab') as fout:
            fout.write(fin.read())
        return True
    except:
        logger.error("Fail to download {0}. Error: {1}".format(url, sys.exc_info()))
        return False


def _unzip_file(file_path, target_dir):
    logger.info("Unzipping {0} to {1} ...".format(file_path, target_dir))
    try:
        import zipfile
        with zipfile.ZipFile(file_path) as zip_file:
            if os.path.isdir(target_dir):
                pass
            else:
                os.makedirs(target_dir)
            for names in zip_file.namelist():
                zip_file.extract(names, target_dir)
        return True
    except:
        logger.error("Fail to unzip. Error: ", sys.exc_info())
        return False


def _extract_tar(file_path, target_dir):
    logger.info("Extracting {0} to {1} ...".format(file_path, target_dir))
    try:
        import tarfile
        with tarfile.open(file_path) as tar:
            tar.extractall(path=target_dir)
    except:
        logger.error("Fail to extract. Error: ", sys.exc_info())
        return False
    return True


def _version_compare(ver1, ver2):
    to_version = lambda ver: tuple([int(x) for x in ver.split('.') if x.isdigit()])
    return to_version(ver1) <= to_version(ver2)


def _get_cntk_version(cntk_root):
    logger.debug("In _get_cntk_version(), cntk_root: {0}".format(cntk_root))
    version = ''
    version_file = os.path.join(cntk_root, "cntk", "version.txt")

    if os.path.isfile(version_file):
        with open(version_file) as fin:
            version = fin.readline().strip()
    logger.debug("In _get_cntk_version(), find cntk_version: {0}".format(version))
    return version


def _update_pathenv_win(path, add):
    path_value = _registry_read(winreg.HKEY_CURRENT_USER, "Environment", "PATH")
    logger.debug("Before update, PATH: {0}".format(path_value))

    if add:
        if path in path_value:
            return
        path_value = path + ";" + path_value
        os.environ["PATH"] = path + ";" + os.environ.get("PATH", "")
    else:
        path_value = path_value.replace(path + ";", "")
        os.environ["PATH"] = os.environ["PATH"].replace(path + ";", "")
    _registry_write(winreg.HKEY_CURRENT_USER, "Environment", "PATH", path_value)


def detect_os():
    # logger.info("Begin to detect OS ...")
    os_name = platform.platform(terse=True)
    os_bit = platform.architecture()[0]
    is_64bit = (os_bit == "64bit")

    logger.info("OS: {0}, {1}".format(os_name, os_bit))

    if (os_name.startswith("Windows")):
        sys_info["OS"] = TOOLSFORAI_OS_WIN
        if not os_name.startswith("Windows-10"):
            logger.warning(
                "We recommend Windows 10 as the primary development OS, other Windows versions are not fully supported.")
    elif (os_name.startswith("Linux")):
        sys_info["OS"] = TOOLSFORAI_OS_LINUX
    elif (os_name.startswith("Darwin")):
        sys_info["OS"] = TOOLSFORAI_OS_MACOS
        is_64bit = sys.maxsize > 2 ** 32
    else:
        logger.error("Your OS({0}-{1}) can't be supported! Only Windows, Linux and MacOS can be supported now.".format(os_name, os_bit))
        return False
    if not is_64bit:
        logger.error("Your OS is not 64-bit OS. Now only 64-bit OS is supported.")
        return False
    return True


def detect_gpu():
    # logger.info("Begin to detect NVIDIA GPU ...")
    gpu_detector_name = 'gpu_detector_' + sys_info["OS"]
    if (sys_info["OS"] == TOOLSFORAI_OS_WIN):
        gpu_detector_name = gpu_detector_name + '.exe'
    gpu_detector_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), gpu_detector_name)

    if not (os.path.isfile(gpu_detector_path)):
        logger.error(
            'Not find GPU detector. Please make sure {0} is in the same directory with the installer script.'.format(
                gpu_detector_name))
        return False
    sys_info["GPU"], return_stdout = _run_cmd(gpu_detector_path, return_stdout=True)
    if not sys_info["GPU"]:
        return_stdout = 'None'
    logger.info('NVIDIA GPU: {0}'.format(return_stdout))
    return True


def detect_vs():
    # logger.info("Begin to detect Visual Studio version...")
    vs = []
    vs_2015_path = _registry_read(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\Microsoft\VisualStudio\14.0",
                                  "InstallDir")
    if (vs_2015_path and os.path.isfile(os.path.join(vs_2015_path, "devenv.exe"))):
        vs.append("VS2015")
    vs_2017_path = _registry_read(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\Microsoft\VisualStudio\SxS\VS7",
                                  "15.0")
    if (vs_2017_path and os.path.isfile(os.path.sep.join([vs_2017_path, "Common7", "IDE", "devenv.exe"]))):
        vs.append("VS2017")
    if (len(vs) == 0):
        logger.warning("Not detect Visual Studio 2017 or 2015! We recommend Visual Studio 2017, "
                       "please manually download and install Visual Studio 2017 form https://www.visualstudio.com/downloads/.")
    else:
        logger.info("Visual Studio: {0}".format(" ".join(vs)))


def detect_python_version():
    # logger.info("Begin to detect python version ...")
    py_architecture = platform.architecture()[0]
    py_version = ".".join(map(str, sys.version_info[0:2]))
    py_full_version = ".".join(map(str, sys.version_info[0:3]))
    sys_info["python"] = py_version.replace('.', '')
    logger.debug("In detect_python_version(), sys_info['python']: {0}".format(sys_info["python"]))
    logger.info("PYTHON: {0}, {1}".format(py_full_version, py_architecture))
    if not (_version_compare("3.5", py_version) and py_architecture == '64bit'):
        logger.error("64-bit PYTHON 3.5 or higher is required to run this installer."
                     " We recommend latest PYTHON 3.5 (https://www.python.org/downloads/release/python-355/).")
        return False
    return True


def detect_tf_version():
    logger.info("Begin to detect tensorflow version ...")
    try:
        import tensorflow as tf
        logger.debug("Import tensorflow successfully!")
        tf_version = tf.__version__
        sys_info["tensorflow"] = tf_version
        logger.info("Detect tensorflow version information: {0}".format(tf_version))
    except ImportError:
        logger.error("Import tensorflow failed! Please manually check the installation of tensorflow.")
    except:
        logger.error("Unexpected error: {0}, please manually check the installation of tensorflow.".format(sys.exc_info()[0]))


# def detect_cuda():
#     if (sys_info["OS"] == TOOLSFORAI_OS_WIN):
#         return detect_cuda_win()
#     return True

def detect_cuda():
    if (sys_info["OS"] == TOOLSFORAI_OS_WIN or sys_info["OS"] == TOOLSFORAI_OS_LINUX):
        return detect_cuda_()
    return True

# def detect_cuda_win():
#     # logger.info("Begin to detect cuda version on Windows ...")
#     status, stdout = _run_cmd("nvcc", ["-V"], True)
#     if status and re.search(r"release\s*8.0,\s*V8.0", stdout):
#         sys_info["CUDA"] = "8.0"
#         logger.info("Cuda: {0}".format(sys_info["CUDA"]))
#         if sys_info["cuda80"]:
#             logger.warning("Detect parameter '--cuda80', the install script will be forced to install dependency package for cuda 8.0.")
#         else:
#             logger.warning("We recommend cuda 9.0 (https://developer.nvidia.com/cuda-toolkit)."
#                            "If you want to install dependency package for cuda 8.0, please run the install script with '--cuda80' again.")
#             return False
#     elif status and re.search(r"release\s*9.0,\s*V9.0", stdout):
#         sys_info["CUDA"] = "9.0"
#         logger.info("Cuda: {0}".format(sys_info["CUDA"]))
#         if sys_info["cuda80"]:
#             sys_info["CUDA"] = "8.0"
#             logger.warning("Detect parameter '--cuda80', the install script will be forced to install dependency package for cuda 8.0.")
#     else:
#         sys_info["CUDA"] = "9.0"
#         logger.warning("Not detect cuda! We recommend cuda 9.0 (https://developer.nvidia.com/cuda-toolkit). "
#                        "The install script will install dependency package for cuda 9.0 by default.")
#     return True

def detect_cuda_():
    status, stdout = _run_cmd("nvcc", ["-V"], True)
    if status and re.search(r"release\s*8.0,\s*V8.0", stdout):
        sys_info["CUDA"] = "8.0"
        logger.info("CUDA: {0}".format(sys_info["CUDA"]))
        if sys_info["cuda80"]:
            logger.warning("Detect parameter '--cuda80', the installer script will be forced to install dependency package for CUDA 8.0.")
            return True
        else:
            logger.warning("We recommend CUDA 9.0 (https://developer.nvidia.com/cuda-toolkit)."
                           "If you want to install dependency package for CUDA 8.0, please run the installer script with '--cuda80' again.")
            return False
    elif status and re.search(r"release\s*9.0,\s*V9.0", stdout):
        sys_info["CUDA"] = "9.0"
        logger.info("CUDA: {0}".format(sys_info["CUDA"]))
    else:
        sys_info["CUDA"] = "9.0"
        logger.warning("Not detect CUDA! We recommend CUDA 9.0 (https://developer.nvidia.com/cuda-toolkit). "
                       "The installer script will install dependency package for CUDA 9.0 by default.")
    if sys_info["cuda80"]:
        sys_info["CUDA"] = "8.0"
        logger.warning(
            "Detect parameter '--cuda80', the installer script will be forced to install dependency package for CUDA 8.0.")
    return True

def detect_cudnn():
    if (sys_info["OS"] == TOOLSFORAI_OS_WIN):
        detect_cudnn_win()


def detect_cudnn_win():
    # logger.info("Begin to detect cudnn version on Windows ...")
    if sys_info["CUDA"] == "8.0":
        required_cndunn = {'6': 'cudnn64_6.dll', '7': 'cudnn64_7.dll'}
    else:
        required_cndunn = {'7': 'cudnn64_7.dll'}
    cmd = r"C:\Windows\System32\where.exe"
    for version, dll in required_cndunn.items():
        args = [dll]
        status, cudnn = _run_cmd(cmd, args, True)
        if status and next(filter(os.path.isfile, cudnn.split('\n')), None):
            sys_info["cudnn"] = version
            logger.info("Cudnn: {0}".format(version))
    if not sys_info["cudnn"]:
        logger.warning("Not detect Cudnn! We recommand cudnn 7, please download and install Cudnn 7 from https://developer.nvidia.com/rdp/cudnn-download.")

def detect_mpi_win():
    # logger.info("Begin to detect MPI version on Windows ...")
    target_version = "7.0.12437.6"

    mpi_path = _registry_read(winreg.HKEY_LOCAL_MACHINE, r"Software\Microsoft\MPI", "InstallRoot")
    if (mpi_path and os.path.isfile(os.path.sep.join([mpi_path, "bin", "mpiexec.exe"]))):
        sys_info["mpi"] = _registry_read(winreg.HKEY_LOCAL_MACHINE, r"Software\Microsoft\MPI", "Version")
    if sys_info["mpi"]:
        logger.info("MPI: {0}".format(sys_info["mpi"]))
        if not _version_compare(target_version, sys_info["mpi"]):
            logger.warning("CNTK suggests MPI version to be {0}, please manually upgrade MPI.".format(target_version))
            return False
        return True
    else:
        logger.warning("Not detect MPI, please manually download and isntall MPI.")
        return False


def detect_visualcpp_runtime_win():
    # logger.info("Begin to detect Visuall C++ runtime ...")
    pattern = re.compile(
        "(^Microsoft Visual C\+\+ 201(5|7) x64 Additional Runtime)|(^Microsoft Visual C\+\+ 201(5|7) x64 Minimum Runtime)")
    items = [(winreg.HKEY_CURRENT_USER, r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall"),
             (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall"),
             (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall")]
    for hkey, keypath in items:
        try:
            current_key = winreg.OpenKey(hkey, keypath)
            for subkey in _registry_subkeys(hkey, keypath):
                display_name = _registry_read(current_key, subkey, "DisplayName")
                if (display_name and pattern.match(display_name)):
                    logger.info("Detect Visual C++ runtime already installed.")
                    return True
            winreg.CloseKey(current_key)
        except WindowsError:
            pass
    logger.warning("Not detect Visual C++ runtime.")
    return False

def detect_git():
    res = _run_cmd("git", ["--version"])
    sys_info["git"] = res
    if res:
        logger.info("Git: {0}".format(res))
    else:
        logger.info("Git: {0} (Git is needed, otherwise some dependency packages can't be installed.)".format(res))

def install_cntk(target_dir):
    logger.info("Begin to install CNTK(BrainScript) ...")
    if sys_info["OS"] != TOOLSFORAI_OS_WIN and sys_info["OS"] != TOOLSFORAI_OS_LINUX:
        logger.warning("CNTK(BrainScript) is not supported on your OS, we recommend 64-bit Windows-10 OS or 64-bit Linux OS.")
        # fail_install.append("CNTK(BrainScript)")
        return False
    if sys_info["CUDA"] == "8.0":
        ver = "2.3.1"
    else:
        ver = "2.5.1"
    target_version = 'CNTK-{0}'.format(ver.replace('.', '-'))
    logger.debug("In install_cntk(), target_version: {0}".format(target_version))
    version = _get_cntk_version(target_dir)
    if target_version == version:
        logger.info('CNTK(BrainScript)-{0} is already installed.'.format(ver))
        return True
    logger.debug('In install_cntk(), target_dir: {0}'.format(target_dir))
    cntk_root = os.path.join(target_dir, 'cntk')
    if os.path.isdir(cntk_root):
        try:
            shutil.rmtree(cntk_root)
        except:
            logger.error('Fail to install CNTK(BrainScript), the error message: can not remove old version in directory {0}.'
                         'Please manually remove old version, and run the installer script again.'.format(cntk_root))
            # fail_install.append("CNTK(BrainScript)")
            return False
    if not os.path.isdir(target_dir):
        try:
            os.makedirs(target_dir)
        except:
            logger.error('Fail to install CNTK(BrainScript), the error message: can not create directory {0}.'
                         'Please check if there is permission for creating directory.'.format(target_dir))
            # fail_install.append("CNTK(BrainScript)")
            return False
    cntk_file_name = "{}-{}-64bit-{}.{}".format(target_version, "Windows" if sys_info["OS"] == TOOLSFORAI_OS_WIN else "Linux",
                                                "GPU" if sys_info["GPU"] else "CPU-Only", "zip" if sys_info["OS"] == TOOLSFORAI_OS_WIN else "tar.gz")
    logger.debug("In install_cntk(), cntk_file_name: {0}".format(cntk_file_name))
    cntk_url = "https://cntk.ai/BinaryDrop/{0}".format(cntk_file_name)
    logger.debug("In install_cntk(), cntk_url: {0}".format(cntk_url))
    cntk_file_path = os.path.join(target_dir, cntk_file_name)
    logger.debug("In install_cntk(), cntk_file_path: {0}".format(cntk_file_path))

    if sys_info["OS"] == TOOLSFORAI_OS_WIN:
        download_dir = cntk_file_path
    elif sys_info["OS"] == TOOLSFORAI_OS_LINUX:
        download_dir = os.path.join(r"/tmp", cntk_file_name)
    skip_downloading = False
    if not skip_downloading:
        if not _download_file(cntk_url, download_dir):
            logger.error('Fail to install CNTK(BrainScript), the error message: cannot download {0}.'
                         'Please check your network.'.format(cntk_url))
            # fail_install.append("CNTK(BrainScript)")
            return False

    if (not (_unzip_file(download_dir, target_dir) if sys_info["OS"] == TOOLSFORAI_OS_WIN else _extract_tar(download_dir, target_dir))):
        logger.error('Fail to install CNTK(BrainScript), the error message: cannot decompress the downloaded package.')
        # fail_install.append("CNTK(BrainScript)")
        return False

    if not skip_downloading:
        if os.path.isfile(download_dir):
            os.remove(download_dir)

    if (sys_info["OS"] == TOOLSFORAI_OS_WIN):
        suc = install_cntk_win(cntk_root)
    else:
        suc = install_cntk_linux(cntk_root)

    version = _get_cntk_version(target_dir)
    if (suc and (target_version == version)):
        logger.info("Install CNTK(BrainScript) successfully!")
        logger.warning("Please open a new terminal to make the updated Path environment variable effective.")
        return True
    else:
        logger.error("Fail to install CNTK(BrainScript).")
        logger.warning("Please manually install {0} and update PATH environment.".format(target_version))
        logger.warning("You can reference this link based on your OS: https://docs.microsoft.com/en-us/cognitive-toolkit/Setup-CNTK-on-your-machine")
        # fail_install.append("CNTK(BrainScript)")
        return False
    return True


def install_cntk_linux(cntk_root):
    logger.warning("CNTK(BrainScript) V2 on Linux requires C++ Compiler and Open MPI. "
                   "Please refer to https://docs.microsoft.com/en-us/cognitive-toolkit/Setup-Linux-Binary-Manual")
    bashrc_file_path = os.path.sep.join([os.path.expanduser('~'), '.bashrc'])
    content = ''
    with open(bashrc_file_path, 'r') as bashrc_file:
        content = bashrc_file.read()

    with open(bashrc_file_path, 'a+') as bashrc_file:
        CNTK_PATH = '{0}/cntk/bin'.format(cntk_root)
        CNTK_PATH_EXPORT = 'export PATH={0}:$PATH'.format(CNTK_PATH)
        if not (CNTK_PATH_EXPORT in content):
            bashrc_file.write('{0}\n'.format(CNTK_PATH_EXPORT))

        CNTK_LD_LIBRARY_PATH = '{0}/cntk/lib:{0}/cntk/dependencies/lib'.format(cntk_root)
        CNTK_LD_LIBRARY_PATH_EXPORT = 'export LD_LIBRARY_PATH={0}:$LD_LIBRARY_PATH'.format(CNTK_LD_LIBRARY_PATH)
        if not (CNTK_LD_LIBRARY_PATH_EXPORT in content):
            bashrc_file.write('{0}\n'.format(CNTK_LD_LIBRARY_PATH_EXPORT))

    return True


def install_cntk_win(cntk_root):
    suc = True
    try:
        _update_pathenv_win(os.path.join(cntk_root, "cntk"), True)
        if (not detect_mpi_win()):
            mpi_exe = os.path.sep.join([cntk_root, "prerequisites", "MSMpiSetup.exe"])
            logger.debug("MPI exe path: %s" % mpi_exe)
            logger.info("Begin to install MPI ...")
            _run_cmd_admin(mpi_exe, "-unattend")
            if (detect_mpi_win()):
                logger.info("Install MPI successfully.")
            else:
                suc = False
                logger.error("Fail to install MPI. Please manually install MPI >= 7.0.12437.6")

        if (not detect_visualcpp_runtime_win()):
            vc_redist_exe = os.path.sep.join([cntk_root, "prerequisites", "VS2015", "vc_redist.x64.exe"])
            logger.debug("VC redist exe path: {0}".format(vc_redist_exe))
            logger.info("Begin to install Visual C++ runtime ...")
            _run_cmd_admin(vc_redist_exe, "/install /norestart /passive")
            if (detect_visualcpp_runtime_win()):
                logger.info("Install Visual C++ runtime successfully.")
                logger.warning(" Please manually install Visual C++ Redistributable Package for Visual Studio 2015 or 2017.")
            else:
                suc = False
                logger.error("Fail to install Visual C++ runtime.")
    except:
        suc = False
        logger.error("Fail to install CNTK(BrainScript). The error massage: {0}".format(sys.exc_info()))

    # if "AITOOLS_CNTK_ROOT" in os.environ:
    #     logger.debug("Delete environment variable: AITOOLS_CNTK_ROOT")
    #     _registry_delete(winreg.HKEY_CURRENT_USER, "Environment", "AITOOLS_CNTK_ROOT")

    #if (_run_cmd("SETX", ["AITOOLS_CNTK_ROOT", cntk_root])):
    #    logger.debug("Set CNTK(BrainScript) root path successfully.")
    #else:
    #    logger.debug("Fail to set CNTK(BrainScript) root path.")

    return suc

def delete_env(name):
    try:
        logger.debug("Delete environment variable: {0}".format(name))
        return _registry_delete(winreg.HKEY_CURRENT_USER, "Environment", name)
    except:
        logger.debug("Environment variable {0} doesn't exist.".format(name))
        return True


def pip_install_package(name, options, version="", pkg=None):
    try:
        logger.info("Begin to pip-install {0} {1} ...".format(name, version))
        if not pkg:
            if version:
                if version.strip()[0] == "<" or version.strip()[0] == ">":
                    pkg = "{0}{1}".format(name, version)
                else:
                    pkg = "{0} == {1}".format(name, version)
            else:
                pkg = name
        logger.debug("pkg : {0}".format(pkg))
        res = -1
        # res = pip.main(["install", *options, pkg])
        res = subprocess.check_call([sys.executable, '-m', 'pip', 'install', *options, "-q", pkg])
        if res != 0:
            logger.error("Fail to pip-install {0}.".format(name))
            fail_install.append("%s %s" % (name, version))
        else:
            logger.info("Pip-install {0} {1} successfully!".format(name, version))
        return res == 0
    except Exception as e:
        # logger.error("Fail to pip-install {0}, unexpected error: {0}".format(name, e))
        logger.error("Fail to pip-install {0}, unexpected error! Please try to run installer script again!".format(name))
        fail_install.append("%s %s" % (name, version))
        return False


def pip_uninstall_packge(name, options, version=""):
    try:
        logger.info("Begin to pip-uninstall {0} {1} ...".format(name, version))
        options_copy = options.copy()
        if len(options_copy) != 0 and options_copy[0] == "--user":
            options_copy.pop(0)
        res = -1
        # res = pip.main(["uninstall", *options, name])
        res = subprocess.check_call([sys.executable, '-m', 'pip', 'uninstall', *options_copy, "-y", "-q", name])
        if res != 0:
            logger.error("Fail to pip-uninstall {0}.".format(name))
        else:
            logger.info("Pip-uninstall {0} {1} successfully!".format(name, version))
        return res == 0
    except Exception as e:
        # logger.error("Fail to pip-uninstall {0}, unexpected error: {1}".format(name, e))
        logger.error("Fail to pip-uninstall {0}, unexpected error! Please try to run installer script again!".format(name))
        return False


def pip_install_scipy(options):
    logger.info("Begin to install scipy(numpy, scipy) ...")
    name = "numpy"
    version = "1.14.2"
    if not pip_install_package(name, options, version):
        logger.error("Pip_install_scipy terminated due to numpy installation failure.")
        return False

    name = "scipy"
    version = "1.0.1"
    if not pip_install_package(name, options, version):
        logger.error("Pip_install_scipy terminated due to scipy installation failure.")
        return False
    return True


def pip_install_tensorflow(options):
    name = "tensorflow%s" % ("-gpu" if sys_info["GPU"] else "")
    if sys_info["CUDA"] == "8.0":
        if sys_info["OS"] == TOOLSFORAI_OS_WIN:
            version = "1.4.0"
        elif sys_info["OS"] == TOOLSFORAI_OS_LINUX:
            version = "1.4.1"
    else:
        version = "1.5.0"
    return pip_install_package(name, options, version)


def pip_install_pytorch(options):
    name = "torch"
    version = "0.4.0"
    if sys_info["OS"] == TOOLSFORAI_OS_MACOS:
        pip_install_package(name, options, version)
    elif sys_info["OS"] == TOOLSFORAI_OS_WIN or sys_info["OS"] == TOOLSFORAI_OS_LINUX:
        wheel_ver = sys_info["python"]
        arch = "win_amd64" if sys_info["OS"] == TOOLSFORAI_OS_WIN else "linux_x86_64"
        if not sys_info["GPU"]:
            gpu_type = "cpu"
        else:
            gpu_type = "cu80" if sys_info["CUDA"] == "8.0" else "cu90"
        pkg = "http://download.pytorch.org/whl/{0}/{1}-{2}-cp{3}-cp{3}m-{4}.whl".format(gpu_type, name, version,
                                                                                        wheel_ver, arch)
        pip_install_package(name, options, version, pkg)
    else:
        logger.error("Fail to install pytorch.")
        logger.warning("Pytorch installation can not be supported on your OS! We recommand 64-bit Windows-10, Linux and Macos.")

    name = "torchvision"
    version = ""
    pip_install_package(name, options)

def pip_install_cntk(options):
    if not ((sys_info["OS"] == TOOLSFORAI_OS_WIN) or (sys_info["OS"] == TOOLSFORAI_OS_LINUX)):
        logger.info("CNTK(Python) can not be supported on your OS, we recommend 64-bit Windows-10 OS or 64-bit Linux OS.")
        return
    name = ""
    if sys_info["GPU"]:
        name = "cntk-gpu"
    else:
        name = "cntk"
    if sys_info["CUDA"] == "8.0":
        version = "2.3.1"
        wheel_ver = sys_info["python"]
        arch =  "win_amd64" if sys_info["OS"] == TOOLSFORAI_OS_WIN else "linux_x86_64"
        gpu_type = "GPU" if sys_info["GPU"] else "CPU-Only"
        pkg = "https://cntk.ai/PythonWheel/{0}/cntk-{1}-cp{2}-cp{2}m-{3}.whl".format(gpu_type, version, wheel_ver, arch)
        return pip_install_package(name, options, version, pkg)
    else:
        version = "2.5.1"
        return pip_install_package(name, options, version)

def pip_install_keras(options):
    name = "Keras"
    version = "2.1.5"
    return pip_install_package(name, options, version)


def pip_install_caffe2(options):
    if not (sys_info["OS"] == TOOLSFORAI_OS_WIN):
        logger.warning("In non-Windows OS, you need to manually install caffe2 from source.")
        return
    name = "caffe2"
    version = "0.8.1"
    arch = "win_amd64"
    wheel_ver = sys_info["python"]
    if sys_info["GPU"] and sys_info["CUDA"] == "8.0":
        pkg = "https://raw.githubusercontent.com/linmajia/ai-package/master/caffe2/{0}/caffe2_gpu-{0}-cp{1}-cp{1}m-{2}.whl".format(
            version, wheel_ver, arch)
    else:
        pkg = "https://raw.githubusercontent.com/linmajia/ai-package/master/caffe2/{0}/caffe2-{0}-cp{1}-cp{1}m-{2}.whl".format(
            version, wheel_ver, arch)
    return pip_install_package(name, options, version, pkg)


def pip_install_theano(options):
    name = "Theano"
    version = "1.0.1"
    return pip_install_package(name, options, version)


def pip_install_mxnet(options):
    version = "1.1.0.post0"
    name = ""
    if sys_info["GPU"]:
        if sys_info["CUDA"] == "8.0":
            name = "mxnet-cu80"
        else:
            name = "mxnet-cu90"
    else:
        name = "mxnet"

    return pip_install_package(name, options, version)


def pip_install_chainer(options):
    # cupy installation for GPU linux
    logger.info("Begin to install chainer(cupy, chainer, chainermn) ...")
    name = "cupy"
    version = "4.0.0"
    if (sys_info["GPU"] and (sys_info["OS"] == TOOLSFORAI_OS_LINUX)):
        # logger.info("Install cupy to support CUDA for chainer.")
        if sys_info["CUDA"] == "8.0":
            name = "cupy-cuda80"
        elif sys_info["CUDA"] == "9.0":
            name = "cupy-cuda90"
        pip_install_package(name, options)
    elif (sys_info["GPU"] and (sys_info["OS"] == TOOLSFORAI_OS_WIN)):
        try:
            cupy = importlib.import_module(name)
            if (not _version_compare(version, cupy.__version__)):
                logger.warning("Cupy's version is too low, please manually upgrade cupy >= 2.0.0.")
            else:
                logger.info("Cupy is already installed.")
        except ImportError:
            logger.warning("On windows, please manully install cupy. You can reference this link https://github.com/Microsoft/vs-tools-for-ai/blob/master/docs/prepare-localmachine.md#chainer.")

    name = "chainer"
    version = "4.0.0"
    pip_install_package(name, options, version)

    name = "chainermn"
    version = ""
    if not pip_install_package(name, options):
        logger.warning("On Linux, in order to install chainermn, please manually install libmpich-dev and run installer script again.")

def pip_install_onnxmltools(options):
    name = "onnxmltools"
    version = ""
    if module_exists(name):
        logger.info("{0} is already installed.".format(name))
    else:
        pip_install_package(name, options)

# converter related
def pip_install_winmltools(options):
    name = "winmltools"
    version = ""
    if module_exists(name):
        logger.info("{0} is already installed.".format(name))
    else:
        pip_install_package(name, options)


def pip_install_coremltools(options):
    name = "coremltools"
    version = ""
    if sys_info["OS"] == TOOLSFORAI_OS_WIN:
        if sys_info["git"]:
            pkg = "git+https://github.com/apple/coremltools@v0.8"
            return pip_install_package(name, options, version, pkg)
        else:
            fail_install.append("%s %s" % (name, version))
            logger.warning("Fail to install {0}. Please manually install git and run installer script again.".format(name))
            return False
    else:
        return pip_install_package(name, options)

def pip_install_onnx(options):
    name = "onnx"
    version = "1.1.2"
    return pip_install_package(name, options, version)


def pip_install_tf2onnx(options):
    name = "tf2onnx"
    version = "0.0.0.1"
    if not sys_info["git"]:
        fail_install.append("%s %s" % (name, version))
        logger.warning("Fail to install {0}. Please manually install git and run installer script again.".format(name))
        return False

    pkg = "git+https://github.com/onnx/tensorflow-onnx.git@r0.1"
    if module_exists(name):
        logger.info("{0} is already installed. We will uninstall it and upgrade to the latest version.".format(name))
        pip_uninstall_packge(name, options, version)

    return pip_install_package(name, options, version, pkg)


def pip_install_extra_software(options):
    logger.info("Begin to install extra software(jupyter, matplotlib, and pandas) ...")
    name = "jupyter"
    version = ""
    if module_exists(name):
        logger.info("{0} is already installed.".format(name))
    else:
        pip_install_package(name, options, version)

    name = "matplotlib"
    version = ""
    if module_exists(name):
        logger.info("{0} is already installed.".format(name))
    else:
        pip_install_package(name, options, version)

    name = "pandas"
    version = ""
    if module_exists(name):
        logger.info("{0} is already installed.".format(name))
    else:
        pip_install_package(name, options, version)


def pip_install_converter(options):
    logger.info("Begin to install converter(coremltools, onnx, tf2onnx, onnxmltools and winmltools) ...")
    try:
        pip_install_coremltools(options)
        pip_install_onnx(options)
        pip_install_tf2onnx(options)
        pip_install_onnxmltools(options)
        pip_install_winmltools(options)
    except Exception as e:
        # logger.error("Fail to install converter, unexpected error: {0}".format(e))
        logger.error("Fail to install converter, unexpected error! Please run installer again!")


def pip_install_ml_software(options):
    logger.info("Begin to install ml software(scikit-learn, xgboost and libsvm) ...")
    name = "scikit-learn"
    version = "0.19.1"
    pip_install_package(name, options, version)

    name = "xgboost"
    version = "0.71"
    if sys_info["OS"] != TOOLSFORAI_OS_WIN:
        logger.warning("In order to install xgboost, C++ compiler is needed.")
        pip_install_package(name, options, version)
    else:
        if sys_info["python"] == "35":
            pkg = "https://raw.githubusercontent.com/linmajia/ai-package/master/xgboost/{0}/xgboost-{0}-cp35-cp35m-win_amd64.whl".format(version)
        elif sys_info["python"] == "36":
            pkg = "https://raw.githubusercontent.com/linmajia/ai-package/master/xgboost/{0}/xgboost-{0}-cp36-cp36m-win_amd64.whl".format(version)
        pip_install_package(name, options, version, pkg)

    name = "libsvm"
    version = "3.22"
    if sys_info["OS"] != TOOLSFORAI_OS_WIN:
        logger.warning(
            "On Linux or Mac, in order to install {0}=={1}, please manually download source code and install it.".format(
                name, version))
        return
    if sys_info["python"] == "35":
        pkg = "https://raw.githubusercontent.com/linmajia/ai-package/master/libsvm/3.22/libsvm-3.22-cp35-cp35m-win_amd64.whl"
    elif sys_info["python"] == "36":
        pkg = "https://raw.githubusercontent.com/linmajia/ai-package/master/libsvm/3.22/libsvm-3.22-cp36-cp36m-win_amd64.whl"
    logger.debug("Pip install libsvm from {0}".format(pkg))
    pip_install_package(name, options, version, pkg)


def pip_software_install(options, user, verbose):
    pip_ops = []
    if options:
        pip_ops = options.split()
    elif user:
        pip_ops = ["--user"]

    if not verbose:
        pip_ops.append("-q")

    if not pip_install_scipy(pip_ops):
        return

    pip_install_cntk(pip_ops)
    pip_install_tensorflow(pip_ops)
    pip_install_pytorch(pip_ops)
    pip_install_mxnet(pip_ops)
    pip_install_chainer(pip_ops)
    pip_install_theano(pip_ops)
    pip_install_keras(pip_ops)
    pip_install_caffe2(pip_ops)
    pip_install_ml_software(pip_ops)
    pip_install_converter(pip_ops)
    pip_install_extra_software(pip_ops)


def set_ownership_as_login(target_dir):
    if (sys_info["OS"] == TOOLSFORAI_OS_WIN):
        return

    try:
        import grp
        import pwd
        import getpass

        if ((not os.path.isdir(target_dir)) or (not os.path.exists(target_dir))):
            return

        real_user = os.getlogin()
        real_group = grp.getgrgid(pwd.getpwnam(real_user).pw_gid).gr_name
        if ((not real_user) or (not real_group)):
            return

        if (real_user != getpass.getuser()):
            _run_cmd('chown', ['-R', '{0}:{1}'.format(real_user, real_group), target_dir])
    except:
        pass


def fix_directory_ownership():
    # On Linux, if users install with "sudo", then ~/.toolsforai will have wrong directory ownership.
    target_dir = os.path.sep.join([os.path.expanduser('~'), '.toolsforai'])
    set_ownership_as_login(target_dir)


logger = _init_logger()

try:
    import pip
except ImportError:
    logger.error("you need to install pip first.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", help="give more output to debug log level.", action="store_true")
    parser.add_argument("-u", "--user", help="install to the Python user install directory for your platform.", action="store_true")
    parser.add_argument("--cuda80", help="forcing the installation of the dependency packages for cuda 8.0.", action="store_true")
    parser.add_argument("-o", "--options", help="add extra options for packages installation. --user ignored if this option is supplied.")
    args, unknown = parser.parse_known_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    if args.cuda80:
        sys_info["cuda80"] = True

    logger.info("Detecting system information ...")
    if not detect_os() or not detect_python_version() or not detect_gpu():
        return
    detect_git()
    if (sys_info["OS"] == TOOLSFORAI_OS_WIN):
       detect_vs()
    if (sys_info["GPU"]):
        if not detect_cuda():
            return
        detect_cudnn()

    target_dir = ''
    if sys_info["OS"] == TOOLSFORAI_OS_WIN:
        target_dir = os.path.sep.join([os.getenv("APPDATA"), "Microsoft", "ToolsForAI", "RuntimeSDK"])
    elif sys_info["OS"] == TOOLSFORAI_OS_LINUX:
        target_dir = os.path.sep.join([os.path.expanduser('~'), '.toolsforai', 'RuntimeSDK'])

    try:
        _thread.start_new_thread(install_cntk, (target_dir, ))
    except:
        logger.error("Fail to startup install_cntk thread!")

    pip_software_install(args.options, args.user, args.verbose)
    delete_env("AITOOLS_CNTK_ROOT")
    fix_directory_ownership()
    for pkg in fail_install:
        logger.info("Fail to install {0}. Please try to run installer script again!".format(pkg))
    logger.info('Setup finishes.')
    input('Press enter to exit.')


if __name__ == "__main__":
    main()
