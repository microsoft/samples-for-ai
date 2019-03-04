using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using GalaSoft.MvvmLight;
using GalaSoft.MvvmLight.Command;
using PictureStyleTransfer.Camera;
using PictureStyleTransferLibrary;
using System;
using System.Collections.Generic;
using System.Configuration;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Threading;

namespace PictureStyleTransfer.ViewModel
{
    public class MainViewModel : ViewModelBase
    {
        /// <summary>
        /// Initializes a new instance of the MainViewModel class.
        /// </summary>
        public MainViewModel()
        {
            TakeFrame = new RelayCommand(ExecuteTakeFrame, CanTakeFrame);
            InitBinding();
            new Thread(() =>
            {
                InitSetting();
            }).Start();

        }

        private static int WIDTH = 640;
        private static int HEIGHT = 480;

        private TransferFactory _styleFactory;
        private IModel _model ;
        private ImageSource _imageSource;
        private ImageSource _transferredImageSource;
        private ImageSource _frameImageSource;
        private List<CameraDevice> _cameraItems;
        private CameraDevice _selectedCamera;
        private string[] _styleNames;
        private string _selectedStyle;
        private bool _isEnable;
        private WriteableBitmap originBitmap;
        private WriteableBitmap transBitmap ;
        private WriteableBitmap frameBitmap ;

        private Task transferImageTask;

        private volatile bool photograph = false;
        private volatile bool _isTransferring = false;
        private volatile bool _isRunning = true;


        private VideoCapture capture;

        public bool IsRunning
        {
            get { return _isRunning; }
            set
            {
                if (_isRunning == value) return;
                _isRunning = value;
                RaisePropertyChanged();
            }
        }

        public void Stop()
        {
            IsRunning = false;
        }

        public List<CameraDevice> CameraItems
        {
            get { return _cameraItems; }
            set
            {
                if (_cameraItems == (value)) return;
                _cameraItems = value;
                RaisePropertyChanged();
            }
        }

        public CameraDevice SelectedCamera
        {
            get { return _selectedCamera; }
            set
            {
                if (_selectedCamera == value) return;
                _selectedCamera = value;
                if (capture != null)
                {
                    capture.Stop();
                }
                newCameraPlay(originBitmap, transBitmap, frameBitmap);

                RaisePropertyChanged();
            }
        }

        public bool IsEnable
        {
            get { return _isEnable; }
            set
            {
                if (_isEnable == value) return;
                _isEnable = value;
                RaisePropertyChanged();
            }
        }

        public string[] StyleItems
        {
            get { return _styleNames; }
            set
            {
                if (_styleNames == value) return;
                _styleNames = value;
                RaisePropertyChanged();
            }
        }

        public string SelectedStyle
        {
            get { return _selectedStyle; }
            set
            {
                if (_selectedStyle == value) return;
                _selectedStyle = value;
                _model = _styleFactory.GetTransfer(_selectedStyle);
                RaisePropertyChanged();
            }
        }

        public ImageSource OriginImageSource
        {
            get
            {
                return _imageSource;
            }
            set
            {
                if (_imageSource == value) return;
                _imageSource = value;
                RaisePropertyChanged();
            }
        }

        public ImageSource FrameImageSource
        {
            get
            {
                return _frameImageSource;
            }
            set
            {
                if (_frameImageSource == value) return;
                _frameImageSource = value;
                RaisePropertyChanged();
            }
        }

        public ImageSource TransferredImageSource
        {
            get
            {
                return _transferredImageSource;
            }
            set
            {
                if (_transferredImageSource == value) return;
                _transferredImageSource = value;
                RaisePropertyChanged();
            }
        }

        public RelayCommand TakeFrame { get; private set; }

        private void ExecuteTakeFrame()
        {
            photograph = true;
        }

        bool CanTakeFrame()
        {
            if (_isTransferring)
            {
                MessageBox.Show("Transferring the previous Frame. Please wait for a few seconds");
            }
            return ! _isTransferring;
        }

        private void InitCamera()
        {
            CameraItems = CameraDevice.GetCameraDevice().ToList();
        }

        private void InitStyleModel()
        {
            _styleFactory = new TransferFactory();
            StyleItems = _styleFactory.Supported;
            _model = _styleFactory.GetTransfer(StyleItems.FirstOrDefault());
            var t = _model.Transfer(new byte[WIDTH * HEIGHT * 3]);
            IsEnable = true;
        }

        private void InitBinding()
        {
            originBitmap = new WriteableBitmap(WIDTH, HEIGHT, 96.0, 96.0, PixelFormats.Rgb24, null);
            transBitmap = new WriteableBitmap(WIDTH, HEIGHT, 96.0, 96.0, PixelFormats.Rgb24, null);
            frameBitmap = new WriteableBitmap(WIDTH, HEIGHT, 96.0, 96.0, PixelFormats.Rgb24, null);

            OriginImageSource = originBitmap;
            TransferredImageSource = transBitmap;
            FrameImageSource = frameBitmap;
        }


        private void InitSetting()
        {
            InitCamera();
            InitStyleModel();

            try
            {
                string cameraDefault = ConfigurationManager.AppSettings.Get("defaultCamera").ToLower();
                SelectedCamera = CameraItems.Find(_ => _.Name.ToLower().Contains(cameraDefault));
            }
            catch
            {
                SelectedCamera = CameraItems.FirstOrDefault();
            }
            if (SelectedCamera == null)
            {
                SelectedCamera = CameraItems.FirstOrDefault();
            }
            try
            {
                string styleDefault = ConfigurationManager.AppSettings.Get("defaultStyle").ToLower();
                SelectedStyle = StyleItems.ToList().Find(_ => _.ToLower().Contains(styleDefault));
            }
            catch
            {
                SelectedStyle = StyleItems.FirstOrDefault();
            }
            if (SelectedStyle == null)
            {
                SelectedStyle = StyleItems.FirstOrDefault();
            }
        }

        public void Play()
        {
            newCameraPlay(originBitmap, transBitmap, frameBitmap);
        }

        private Mat frameBgr = new Mat();
        private Mat frameResize = new Mat();
        private Mat frameCut;
        private Mat frameFit = new Mat();
        private Mat frameOrigin = new Mat();
        private Mat cleanMat = new Mat(HEIGHT, WIDTH, DepthType.Default, 3);

        private void ProcessFrame(object sender, EventArgs arg)
        {
            if (capture != null && capture.Ptr != IntPtr.Zero)
            {
                capture.Retrieve(frameBgr, 0);

                double cameraHeight = frameBgr.Height;
                double cameraWidth = frameBgr.Width;

                int resizeWidth, resizeHeight;
                if (cameraHeight / HEIGHT > cameraWidth / WIDTH)
                {
                    resizeHeight = Convert.ToInt32(cameraHeight / (cameraWidth / WIDTH));
                    resizeWidth = WIDTH;
                    
                }
                else
                {
                    resizeWidth = Convert.ToInt32(cameraWidth / (cameraHeight / HEIGHT));
                    resizeHeight = HEIGHT;
                }

                CvInvoke.Resize(frameBgr, frameResize, new System.Drawing.Size(resizeWidth, resizeHeight));

                using (frameCut = new Mat(frameResize, new Range(0, HEIGHT), new Range(0, WIDTH)))
                {

                    CvInvoke.CvtColor(frameCut, frameFit, ColorConversion.Bgr2Rgb);

                    try
                    {
                        originBitmap.Dispatcher?.Invoke(new Action(() =>
                        {
                            originBitmap.WritePixels(new Int32Rect(0, 0, WIDTH, HEIGHT), frameFit.DataPointer, WIDTH * HEIGHT * 3, WIDTH * 3);
                        }), DispatcherPriority.Send);
                    }
                    catch
                    { }
                    if (photograph && !_isTransferring)
                    {
                        _isTransferring = true;
                        photograph = false;
                        IsEnable = false;
                        transferImageTask = new Task(() => TransferImage(transBitmap, frameBitmap, frameFit.Clone() ));
                        transferImageTask.Start();
                    }
                }
            }
        }

        private void newCameraPlay(WriteableBitmap originBitmap, WriteableBitmap transBitmap, WriteableBitmap frameBitmap)
        {
            if (SelectedCamera != null)
            {
                capture = new VideoCapture(SelectedCamera.Index);
                capture.ImageGrabbed += ProcessFrame;
                capture.Start();
            }

        }

        public void TransferImage(WriteableBitmap transBitmap, WriteableBitmap frameBitmap, Mat frame)
        {
            // clean the former transferred image
            try
            {
                transBitmap.Dispatcher?.Invoke(new Action(() =>
                {
                    transBitmap.WritePixels(new Int32Rect(0, 0, WIDTH, HEIGHT), cleanMat.DataPointer, WIDTH * HEIGHT * 3, WIDTH * 3);
                }), DispatcherPriority.DataBind);
            }
            catch { }
            try
            { 
                frameBitmap.Dispatcher?.Invoke(new Action(() =>
                {
                    frameBitmap.WritePixels(new Int32Rect(0, 0, WIDTH, HEIGHT), frame.DataPointer, WIDTH * HEIGHT * 3, WIDTH * 3);
                }), DispatcherPriority.DataBind);
            }
            catch { }

            byte[] rgbs = new byte[WIDTH * HEIGHT * 3];
            
            Marshal.Copy(frame.DataPointer, rgbs, 0, WIDTH * HEIGHT * 3);

            byte[] transferred = new byte[rgbs.Length];
            try
            {
                transferred = _model.Transfer(rgbs);
            }
            catch
            {
                MessageBox.Show("Fail to transfer. You can try again");
            }

            try
            {
                transBitmap.Dispatcher?.Invoke(new Action(() =>
                {
                    transBitmap.WritePixels(new Int32Rect(0, 0, WIDTH, HEIGHT), transferred, WIDTH* 3, 0);
                }), DispatcherPriority.DataBind);
            }
            catch {    }

            _isTransferring = false;
            IsEnable = true;
            
        }
    }
}