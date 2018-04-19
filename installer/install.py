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

TOOLSFORAI_OS_WIN = "win"
TOOLSFORAI_OS_LINUX = "linux"
TOOLSFORAI_OS_MACOS = "mac"
sys_info = {
    "OS": None,
    "python": None,
    "GPU": False,
    "CUDA": None,
    "tensorflow": None
}

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
    except:
        logger.debug("read registry key: {0}, value: {1} error".format(keypath, value_name))
        return None


def _registry_write(hkey, keypath, name, value):
    try:
        registry_key = winreg.CreateKeyEx(hkey, keypath)
        winreg.SetValueEx(registry_key, name, 0, winreg.REG_SZ, value)
        winreg.CloseKey(registry_key)
        return True
    except:
        logger.debug("write registry key: {0}, name: {1}, value: {2} error".format(keypath, name, value))
        return False


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
        logger.debug("===== {:^30} =====".format("%s : stdout" % cmd))
        for line in filter(lambda x: x.strip(), p.stdout.split('\n')):
            logger.debug(line)
        logger.debug("===== {:^30} =====".format("%s : stdout end" % cmd))
        logger.debug("===== {:^30} =====".format("%s : stderr" % cmd))
        for line in filter(lambda x: x.strip(), p.stderr.split('\n')):
            logger.debug(line)
        logger.debug("===== {:^30} =====".format("%s : stderr end" % cmd))
    except Exception as e:
        logger.debug("execute command: %s error." % cmd)
        logger.debug(e)
        status = False
        stdout = ""
    if return_stdout:
        return status, stdout
    else:
        return status


def _wait_process(processHandle, timeout=-1):
    try:
        ret = ctypes.windll.kernel32.WaitForSingleObject(processHandle, timeout)
        logger.debug("wait process return value: %d" % ret)
    except:
        logger.debug("wait process error.")
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
        logger.error("run command as admin error. cmd: %s" % cmd)
        logger.error(e)


def _download_file(url, local_path):
    logger.info("Downloading {0} ...".format(url))
    try:
        import urllib.request
        import ssl
        myssl = ssl.create_default_context()
        myssl.check_hostname = False
        myssl.verify_mode = ssl.CERT_NONE
        with urllib.request.urlopen(url, context=myssl) as fin, \
                open(local_path, 'wb') as fout:
            fout.write(fin.read())
        return True
    except:
        logging.error("Fail to download {0}. {1}".format(url, sys.exc_info()))
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
    logger.debug("In _get_cntk_version, cntk_root: {0}".format(cntk_root))
    version = ''
    version_file = os.path.join(cntk_root, "cntk", "version.txt")

    if os.path.isfile(version_file):
        with open(version_file) as fin:
            version = fin.readline().strip()
    logger.debug("In _get_cntk_version, find version: {0}".format(version))
    return version


def _update_pathenv_win(path, add):
    path_value = _registry_read(winreg.HKEY_CURRENT_USER, "Environment", "PATH")
    logger.debug("Before update, PATH : {0}".format(path_value))

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
    os_name = platform.platform(terse=True)
    os_bit = platform.architecture()[0]
    is_64bit = (os_bit == "64bit")

    logger.info("OS: {0}, {1}".format(os_name, os_bit))

    # sys_info["OS"] = None
    if (os_name.startswith("Windows")):
        sys_info["OS"] = TOOLSFORAI_OS_WIN
        if not os_name.startswith("Windows-10"):
            logger.warning(
                "We recommend Windows 10 as the primary development OS, other versions are not fully tested.")
    elif (os_name.startswith("Linux")):
        sys_info["OS"] = TOOLSFORAI_OS_LINUX
    elif (os_name.startswith("Darwin")):
        sys_info["OS"] = TOOLSFORAI_OS_MACOS
        is_64bit = sys.maxsize > 2 ** 32
    else:
        logger.error("Only Windows, macOS and Linux are supported now.")
        return False

    if not is_64bit:
        logger.error("Only 64-bit OS is supported now.")
        return False

    return True


def detect_gpu():
    # sys_info["GPU"] = False
    gpu_detector_name = 'gpu_detector_' + sys_info["OS"]
    if (sys_info["OS"] == TOOLSFORAI_OS_WIN):
        gpu_detector_name = gpu_detector_name + '.exe'
    gpu_detector_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), gpu_detector_name)

    if not (os.path.isfile(gpu_detector_path)):
        logger.error(
            'No GPU detector found. Please make sure {0} is in the same directory with the installer script.'.format(
                gpu_detector_name))
        return False

    sys_info["GPU"], return_stdout = _run_cmd(gpu_detector_path, return_stdout=True)
    if not sys_info["GPU"]:
        return_stdout = 'None'

    logger.info('NVIDIA GPU: {0}'.format(return_stdout))
    return True


def detect_vs():
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
        logger.warning("Please install Visual Studio 2017 or 2015.")
    else:
        logger.info("Installed Visual Studio: {0}".format(" ".join(vs)))


def detect_python_version():
    # sys_info["python"] = None
    py_architecture = platform.architecture()[0]
    py_version = ".".join(map(str, sys.version_info[0:2]))
    py_full_version = ".".join(map(str, sys.version_info[0:3]))
    sys_info["python"] = py_version.replace('.', '')
    logger.debug("python_version: {0}".format(sys_info["python"]))
    logger.info("Python: {0}, {1}".format(py_full_version, py_architecture))
    if not (_version_compare("3.5", py_version) and py_architecture == '64bit'):
        logger.error("64-bit Python 3.5 or higher is required to run this installer."
                     " We recommend latest Python 3.5 (https://www.python.org/downloads/release/python-354/).")
        return False
    return True


def detect_tf_version():
    # sys_info["tensorflow"] = None
    try:
        import tensorflow as tf
        logger.debug("Import tensorflow successfully!")
        tf_version = tf.__version__
        sys_info["tensorflow"] = tf_version
        logger.info("tensorflow_version: {0}".format(tf_version))
    except ImportError:
        logger.error("Import tensorflow failed! Please manually check the installation of tensorflow.")
    except:
        logger.error("Unexpected error: {0}, please manually check the installation of tensorflow.".format(sys.exc_info()[0]))


def detect_cuda():
    # sys_info["CUDA"] = None
    if (sys_info["OS"] == TOOLSFORAI_OS_WIN):
        detect_cuda_win()


def detect_cuda_win():
    status, stdout = _run_cmd("nvcc", ["-V"], True)
    if status and re.search(r"release\s*8.0,\s*V8.0", stdout):
        sys_info["CUDA"] = "8.0"
        logger.warning("CUDA 8.0 found. We recommend CUDA 9.0, otherwise some functions will not work properly.")
    elif status and re.search(r"release\s*9.0,\s*V9.0", stdout):
        sys_info["CUDA"] = "9.0"
        logger.info["CUDA 9.0 found"]
    else:
        logger.warning("CUDA 9.0 is required. Could not find NVIDIA CUDA Toolkit 9.0. "
                       "Please Download and install CUDA 9.0 from https://developer.nvidia.com/cuda-toolkit.")


def detect_cudnn():
    if (sys_info["OS"] == TOOLSFORAI_OS_WIN):
        detect_cudnn_win()


def detect_cudnn_win():
    if sys_info["CUDA"] == "8.0":
        required_cndunn = {'6': 'cudnn64_6.dll', '7': 'cudnn64_7.dll'}
    else:
        required_cndunn = {'7': 'cudnn64_7.dll'}
    cmd = r"C:\Windows\System32\where.exe"
    for version, dll in required_cndunn.items():
        args = [dll]
        status, cudnn = _run_cmd(cmd, args, True)
        if status and next(filter(os.path.isfile, cudnn.split('\n')), None):
            logger.info("cuDNN {0} found".format(version))
        else:
            logger.warning("cuDNN {0} is required. "
                           "Could not find cuDNN {0}. Please Download and install cuDNN {0} from https://developer.nvidia.com/rdp/cudnn-download.".format(
                version))


def detect_mpi_win():
    target_version = "7.0.12437.6"
    mpi_path = _registry_read(winreg.HKEY_LOCAL_MACHINE, r"Software\Microsoft\MPI", "InstallRoot")
    mpi_version = None
    if (mpi_path and os.path.isfile(os.path.sep.join([mpi_path, "bin", "mpiexec.exe"]))):
        mpi_version = _registry_read(winreg.HKEY_LOCAL_MACHINE, r"Software\Microsoft\MPI", "Version")
    if (mpi_version and _version_compare(target_version, mpi_version)):
        logger.info("MSMPI with version: {0} already installed.".format(mpi_version))
        return True
    elif mpi_version:
        logger.warning("MSMPI with version: {0} already installed. CNTK suggests MSMPI version to be {1}."
                       " Please manually update MSMPI.".format(mpi_version, target_version))
        return False
    else:
        logger.info("MSMPI not found.")
        return False


def detect_visualcpp_runtime_win():
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
                    logger.info("Visual C++ runtime found.")
                    return True
            winreg.CloseKey(current_key)
        except WindowsError:
            pass
    logger.warning("Visual C++ runtime not found.")
    return False


def install_cntk(target_dir):
    if sys_info["OS"] != TOOLSFORAI_OS_WIN and sys_info["OS"] != TOOLSFORAI_OS_LINUX:
        logger.info("CNTK is not supported on your OS.")
        return
    # Temporarily disable installing CNTK on Linux
    # If 'sudo' is used, the effective and real users don't match.
    # if sys_info["OS"] == TOOLSFORAI_OS_LINUX:
    #    return
    if sys_info["CUDA"] == "8.0":
        ver = "2.3.1"
    else:
        ver = "2.5"
    target_version = 'CNTK-{0}'.format(ver.replace('.', '-'))
    logger.debug("target_version: {0}".format(target_version))
    version = _get_cntk_version(target_dir)
    if target_version == version:
        logger.info('CNTK {0} already installed'.format(ver))
        return
    logger.debug('CNTK target dir: {0}'.format(target_dir))
    cntk_root = os.path.join(target_dir, 'cntk')
    if os.path.isdir(cntk_root):
        try:
            shutil.rmtree(cntk_root)
        except:
            logger.error('CNTK installation fails: cannot remove old version in directory {0}.'.format(cntk_root))
            return
    if not os.path.isdir(target_dir):
        try:
            os.makedirs(target_dir)
        except:
            logger.error('CNTK installation fails: cannot create directory {0}.'.format(target_dir))
            return
    cntk_file_name = "{}-{}-64bit-{}.{}".format(target_version,
                                                "Windows" if sys_info["OS"] == TOOLSFORAI_OS_WIN else "Linux",
                                                "GPU" if sys_info["GPU"] else "CPU-Only",
                                                "zip" if sys_info["OS"] == TOOLSFORAI_OS_WIN else "tar.gz")
    cntk_file_path = os.path.join(target_dir, cntk_file_name)
    cntk_url = "https://cntk.ai/BinaryDrop/{0}".format(cntk_file_name)

    skip_downloading = False
    if not skip_downloading:
        if not _download_file(cntk_url, cntk_file_path):
            logger.error('CNTK installation fails: cannot download {0}'.format(cntk_url))
            return

    if (not (
    _unzip_file(cntk_file_path, target_dir) if sys_info["OS"] == TOOLSFORAI_OS_WIN else _extract_tar(cntk_file_path,
                                                                                                     target_dir))):
        logger.error('CNTK installation fails: cannot decompress the downloaded package.')
        return

    if not skip_downloading:
        if os.path.isfile(cntk_file_path):
            os.remove(cntk_file_path)

    if (sys_info["OS"] == TOOLSFORAI_OS_WIN):
        suc = install_cntk_win(cntk_root)
    else:
        suc = install_cntk_linux(cntk_root)

    version = _get_cntk_version(target_dir)
    if (suc and (target_version == version)):
        logger.info("CNTK installation succeeds.")
        logger.warning("Please open a new terminal to make the updated Path environment variable effective.")
    else:
        logger.error("CNTK installation fails.")
        logger.warning("Please manually install {0} and update PATH environment.".format(target_version))
        logger.warning(
            "You can reference this link based on your OS: https://docs.microsoft.com/en-us/cognitive-toolkit/Setup-CNTK-on-your-machine")


def install_cntk_linux(cntk_root):
    logger.warning("CNTK V2 on Linux requires C++ Compiler and Open MPI. "
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
            logger.info("Begin MPI installation...")
            _run_cmd_admin(mpi_exe, "-unattend")
            if (detect_mpi_win()):
                logger.info("MPI installation suceeds.")
            else:
                suc = False
                logger.error("MPI installation fails. Please manually install MSMPI >= 7.0.12437.6")

        if (not detect_visualcpp_runtime_win()):
            vc_redist_exe = os.path.sep.join([cntk_root, "prerequisites", "VS2015", "vc_redist.x64.exe"])
            logger.debug("VC redist exe path: {0}".format(vc_redist_exe))
            logger.info("Begin Visual C++ runtime installation...")
            _run_cmd_admin(vc_redist_exe, "/install /norestart /passive")
            if (detect_visualcpp_runtime_win()):
                logger.info("Visual C++ runtime installation suceeds."
                            " Please manually install Visual C++ Redistributable Package for Visual Studio 2015(2017).")
            else:
                suc = False
                logger.error("Visual C++ runtime installation fails.")
    except:
        suc = False
        logger.error("CNTK installation fails.")
        logger.error(sys.exc_info())

    logger.debug("Set cntk root path...")
    if (_run_cmd("SETX", ["AITOOLS_CNTK_ROOT", cntk_root])):
        logger.debug("cntk root path set succeeds.")
    else:
        logger.debug("cntk root path set fails.")

    return suc


def pip_install_package(name, options, version="", pkg=None):
    try:
        logger.info("Begin install {0} {1} ...".format(name, version))
        if not pkg:
            if version:
                pkg = "{0} == {1}".format(name, version)
            else:
                pkg = name
        res = pip.main(["install", *options, pkg])
        if res != 0:
            logger.error("Fail to install {0} pip package.".format(name))
        else:
            logger.info("{0} {1} installed".format(name, version))
        return res == 0
    except Exception as e:
        # print(str(e))
        logger.error(e)
        return False


def pip_uninstall_packge(name, options, version=""):
    try:
        logger.info("Begin uninstall {0} {1} ...".format(name, version))
        res = pip.main(["uninstall", *options, name])
        if res != 0:
            logger.error("Fail to uninstall {0} pip package.".format(name))
        else:
            logger.info("{0} {1} uninstalled successfully.".format(name, version))
        return res == 0
    except:
        return False


def pip_install_scipy(options):
    name = "numpy"
    version = "1.14.2"
    if not pip_install_package(name, options, version):
        logger.error("Pip installation terminated due to numpy installation failure.")
        return False

    name = "scipy"
    version = "1.0.1"
    if not pip_install_package(name, options, version):
        logger.error("Pip installation terminated due to scipy installation failure.")
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
    pip_install_package(name, options, version)


def pip_install_cntk(options):
    if not ((sys_info["OS"] == TOOLSFORAI_OS_WIN) or (sys_info["OS"] == TOOLSFORAI_OS_LINUX)):
        logger.info("cntk pip package not available on your OS.")
        return
    name = "cntk"
    wheel_ver = sys_info["python"]
    arch = "win_amd64" if sys_info["OS"] == TOOLSFORAI_OS_WIN else "linux_x86_64"
    gpu_type = "GPU" if sys_info["GPU"] else "CPU-Only"
    cntk_type = "cntk_gpu" if sys_info["GPU"] else "cntk"
    if sys_info["CUDA"] == "8.0":
        version = "2.3.1"
        pkg =  "https://cntk.ai/PythonWheel/{0}/cntk-{1}-cp{2}-cp{2}m-{3}.whl".format(gpu_type, version, wheel_ver, arch)
    else:
        version = "2.5"
        pkg = "https://cntk.ai/PythonWheel/{0}/{4}-{1}-cp{2}-cp{2}m-{3}.whl".format(gpu_type, version, wheel_ver, arch,
                                                                                cntk_type)
    pip_install_package(name, options, version, pkg)


def pip_install_keras(options):
    name = "Keras"
    version = "2.1.5"
    pip_install_package(name, options, version)


def pip_install_caffe2(options):
    if not (sys_info["OS"] == TOOLSFORAI_OS_WIN):
        logger.warning("You need to install caffe2 from source.")
        return
    name = "caffe2"
    version = "0.8.1"
    arch = "win_amd64"
    wheel_ver = sys_info["python"]
    pkg = "https://github.com/linmajia/ai-package/raw/master/caffe2/{0}/caffe2_gpu-{0}-cp{1}-cp{1}m-{2}.whl".format(
        version, wheel_ver, arch)
    pip_install_package(name, options, version, pkg)


def pip_install_theano(options):
    name = "Theano"
    version = "1.0.1"
    pip_install_package(name, options, version)


def pip_install_mxnet(options):
    version = "1.0.0"
    # if sys_info["CUDA"] == "9.0" and sys_info["OS"] == TOOLSFORAI_OS_WIN:
    #     logger.warning("Mxnet failed to install. In Windows, mxnet {0} don't support for cuda 9.0.".format(version))
    #     return
    if sys_info["GPU"]:
        name = "mxnet%s" % ("-cu90" if sys_info["CUDA"] == "9.0" else "-cu80")
    else:
        name = "mxnet"
    if sys_info["CUDA"] == "9.0" and sys_info["OS"] == TOOLSFORAI_OS_WIN:
        name = "mxnet"
        logger.warning("In windows, mxnet {0} doesn't support cuda 9.0. Instead, we install mxnet for cpu-only".format(version))
    pip_install_package(name, options, version)


def pip_install_chainer(options):
    # cupy installation for GPU linux
    name = "cupy"
    version = "2.5.0"
    if (sys_info["GPU"] and (sys_info["OS"] == TOOLSFORAI_OS_LINUX)):
        logger.info("Install cupy to support CUDA for chainer.")
        pip_install_package(name, options, version)
    elif (sys_info["GPU"] and (sys_info["OS"] == TOOLSFORAI_OS_WIN)):
        try:
            cupy = importlib.import_module(name)
            if (not _version_compare("2.0", cupy.__version__)):
                logger.warning("Please make sure cupy >= 2.0.0 to support CUDA for chainer.")
        except ImportError:
            logger.warning("Please manully install cupy to support CUDA for chainer."
                           "You can reference this link <https://github.com/Microsoft/vs-tools-for-ai/blob/master/docs/prepare-localmachine.md#chainer> to install cupy on Windows")

    name = "chainer"
    version = "3.5.0"
    pip_install_package(name, options, version)


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
    version = "0.8"
    pkg = "git+https://github.com/apple/{0}@v{1}".format(name, version)
    # if module_exists(name):
    #     logger.info("{0} is already installed.".format(name))
    #     return
    pip_install_package(name, options, version, pkg)


def pip_install_onnx(options):
    name = "onnx"
    version = "1.0.1"
    pip_install_package(name, options, version)


def pip_install_tf2onnx(options):
    name = "tf2onnx"
    version = "0.0.0.1"
    pkg = "git+https://github.com/tocean/tensorflow-onnx.git@r0.1"
    if module_exists(name):
        logger.info("{0} is already installed, we will uninstall {0} and reinstall the latest {0}.".format(name))
        pip_uninstall_packge(name, options, version)
    pip_install_package(name, options, version, pkg)


def pip_install_extra_software(options):
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
    try:
        detect_tf_version()
        if ((not sys_info["tensorflow"]) or (not _version_compare("1.5.0", sys_info["tensorflow"]))):
            logger.warning(
                "We recommend tensorflow==1.5.0, otherwise some functions of converter will not work properly.")
        pip_install_coremltools(options)
        pip_install_onnx(options)
        pip_install_tf2onnx(options)
        pip_install_winmltools(options)
    except Exception as e:
        logger.info(e)


def pip_install_ml_software(options):
    name = "scikit-learn"
    version = "0.19.1"
    # if module_exists("sklearn"):
    #     logger.info("{0} is already installed.".format(name))
    # else:
    pip_install_package(name, options, version)

    name = "xgboost"
    version = "0.7"
    if sys_info["OS"] != TOOLSFORAI_OS_WIN:
        logger.warning(
            'In Linux or Mac, You can install {0}=={1} by "pip install ...", and C++ compiler needed.'.format(name,
                                                                                                              version))
        return
    if sys_info["python"] == "35":
        pkg = "https://raw.githubusercontent.com/linmajia/ai-package/master/xgboost/0.7/xgboost-0.7-cp35-cp35m-win_amd64.whl"
    elif sys_info["python"] == "36":
        pkg = "https://raw.githubusercontent.com/linmajia/ai-package/master/xgboost/0.7/xgboost-0.7-cp36-cp36m-win_amd64.whl"
    # if module_exists(name):
    #     logger.info("{0} is already installed.".format(name))
    pip_install_package(name, options, version, pkg)

    name = "libsvm"
    version = "3.22"
    if sys_info["OS"] != TOOLSFORAI_OS_WIN:
        logger.warning(
            "In Linux or Mac, in order to install {0} {1}, please manually download source code and build it.".format(
                name, version))
        return
    if sys_info["python"] == "35":
        pkg = "https://raw.githubusercontent.com/linmajia/ai-package/master/libsvm/3.22/libsvm-3.22-cp35-cp35m-win_amd64.whl"
    elif sys_info["python"] == "36":
        pkg = "https://raw.githubusercontent.com/linmajia/ai-package/master/libsvm/3.22/libsvm-3.22-cp36-cp36m-win_amd64.whl"
    logger.debug("pip install libsvm form {0}".format(pkg))
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
    pip_install_mxnet(pip_ops)
    pip_install_chainer(pip_ops)
    pip_install_theano(pip_ops)
    pip_install_keras(pip_ops)
    if sys_info["CUDA"] == "8.0":
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
    parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
    parser.add_argument("-u", "--user", help="install pip package to user install directory", action="store_true")
    parser.add_argument("--forcecuda8", help="Forced installation of dependency packages for Cuda8.",
                        action="store_true")
    parser.add_argument("-o", "--options",
                        help="pip extra options for installation. --user ignored if this option supplied.")
    args, unknown = parser.parse_known_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    if not detect_os() or not detect_python_version() or not detect_gpu():
        return

    target_dir = ''
    if sys_info["OS"] == TOOLSFORAI_OS_WIN:
        target_dir = os.path.sep.join([os.getenv("APPDATA"), "Microsoft", "ToolsForAI", "RuntimeSDK"])
    elif sys_info["OS"] == TOOLSFORAI_OS_LINUX:
        target_dir = os.path.sep.join([os.path.expanduser('~'), '.toolsforai', 'RuntimeSDK'])

    if (sys_info["OS"] == TOOLSFORAI_OS_WIN):
        detect_vs()

    if (sys_info["GPU"]):
        detect_cuda()
        if args.forcecuda8:
            sys_info["CUDA"] = "8.0"
            logger.warning("Force the installation of the dependency packages for cuda 8.0!")
        detect_cudnn()

    install_cntk(target_dir)
    pip_software_install(args.options, args.user, args.verbose)
    fix_directory_ownership()
    logger.info('Setup finishes.')
    input('Press enter to exit.')


if __name__ == "__main__":
    main()
