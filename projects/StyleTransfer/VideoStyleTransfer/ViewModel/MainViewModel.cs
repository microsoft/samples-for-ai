using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using GalaSoft.MvvmLight;
using System;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Threading;
using System.Drawing;
using GalaSoft.MvvmLight.Command;
using System.Collections.Generic;
using System.Collections.Concurrent;
using VideoStyleTransferLibrary;
using System.Configuration;
using VideoStyleTransfer.Camera;

namespace VideoStyleTransfer.ViewModel
{
    /// <summary>
    /// This class contains properties that the main View can data bind to.
    /// <para>
    /// Use the <strong>mvvminpc</strong> snippet to add bindable properties to this ViewModel.
    /// </para>
    /// <para>
    /// You can also use Blend to data bind with the tool's support.
    /// </para>
    /// <para>
    /// See http://www.galasoft.ch/mvvm
    /// </para>
    /// </summary>
    public class MainViewModel : ViewModelBase
    {
        /// <summary>
        /// Initializes a new instance of the MainViewModel class.
        /// </summary>
        public MainViewModel()
        {
            //ClearOriginFrameBuffer = new RelayCommand(CO);
            //ClearTransFrameBuffer = new RelayCommand(CF);

            IsRunning = true;
            InitImageBinding();
            // the initialization of setting does not bound the UI init 
            new Thread(() =>
            {
                InitSetting();
            }).Start();
            

            _transferTask = new Thread(() => { TranferTask(); });
            _transferTask.Start();

            new Thread(() => { TransferredPlay(_transBitmap); }).Start();
        }

        Thread _transferTask;

        private static int WIDTH = ImageConstants.Width;
        private static int HEIGHT = ImageConstants.Height;

        private TransferFactory _styleFactory = new TransferFactory();
        private IModel _model;
        private string _selectedStyle;
        private string[] _styleNames;

        private ImageSource _imageSource;
        private ImageSource _transferredImageSource;
        private List<CameraDevice> _cameraItems;
        private CameraDevice _selectedCamera;

        private bool _isEnable;
        private WriteableBitmap _originBitmap;
        private WriteableBitmap _transBitmap;

        private const int DEFAULT_FRAME_RATE = 10;

        private int _frameRate = 1;
        private int _batchSize = 1;


        private ConcurrentQueue<KeyValuePair<long, byte[]>> originFrameQueue = new ConcurrentQueue<KeyValuePair<long, byte[]>>();
        private ConcurrentQueue<KeyValuePair<long, byte[]>> transferredFrameQueue = new ConcurrentQueue<KeyValuePair<long, byte[]>>();

        private VideoCapture capture;

        private volatile bool _isRunning = true;



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
            if (_transferTask.ThreadState == ThreadState.Running)
            {
                _transferTask.Abort();
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

        public int BatchSize
        {
            get { return _batchSize; }
            set
            {
                if (_batchSize == value) return;
                _batchSize = value;
                RaisePropertyChanged();
            }
        }

        public int FrameRate
        {
            get { return _frameRate; }
            set
            {
                if (_frameRate == value) return;
                _frameRate = value;
                RaisePropertyChanged();
            }
        }

        private void InitStyleModel()
        {
            StyleItems = _styleFactory.Supported;
        }

        private void InitSetting()
        {
            InitCamera();
            InitStyleModel();

            BatchSize = ImageConstants.Batch;
            try
            {
                FrameRate = Convert.ToInt32(ConfigurationManager.AppSettings.Get("defaultFrameRate"));
            }
            catch
            {
                FrameRate = DEFAULT_FRAME_RATE;
            }
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
            IsEnable = true;
        }

        //public int OriginBufferSize
        //{
        //    get
        //    {
        //        return _originQSize;
        //    }
        //    set
        //    {
        //        if (_originQSize == value) return;
        //        _originQSize = value;
        //        RaisePropertyChanged();
        //    }
        //}

        //public int TransBufferSize
        //{
        //    get
        //    {
        //        return _transQSize;
        //    }
        //    set
        //    {
        //        if (_transQSize == value) return;
        //        _transQSize = value;
        //        RaisePropertyChanged();
        //    }
        //}
        public string SelectedStyle
        {
            get { return _selectedStyle; }
            set
            {
                if (_selectedStyle == value) return;
                IsEnable = false;
                _selectedStyle = value;
                _model = _styleFactory.GetTransfer(_selectedStyle);
                ClearTransBuffer();
                ClearOriginBuffer();
                RaisePropertyChanged();
                IsEnable = true;
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
                IsEnable = false;
                _selectedCamera = value;
                if (capture != null)
                {
                    capture.Stop();
                }
                Play();
                IsEnable = true;
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

        private void InitCamera()
        {
            CameraItems = CameraDevice.GetCameraDevice().ToList();
        }

        private void InitImageBinding()
        {
            _originBitmap = new WriteableBitmap(WIDTH, HEIGHT, 96.0, 96.0, PixelFormats.Rgb24, null);
            _transBitmap = new WriteableBitmap(WIDTH, HEIGHT, 96.0, 96.0, PixelFormats.Rgb24, null);

            OriginImageSource = _originBitmap;
            TransferredImageSource = _transBitmap;
        }

        public void Play()
        {
            newCameraPlay(_originBitmap, _transBitmap);
        }

        private Mat frameBgr = new Mat();
        private Mat frameResize = new Mat();
        private Mat frameCut;
        private Mat frameFit = new Mat();
        private Mat frameOrigin = new Mat();
        private Mat cleanMat = new Mat(HEIGHT, WIDTH, DepthType.Default, 3);

        //public RelayCommand ClearOriginFrameBuffer { get; private set; }
        //public RelayCommand ClearTransFrameBuffer { get; private set; }
        public void ClearOriginBuffer()
        {
            lock (originFrameQueue)
            {
                originFrameQueue = new ConcurrentQueue<KeyValuePair<long, byte[]>>();
            }
        }
        public void ClearTransBuffer()
        {
            lock (transferredFrameQueue)
            {
                transferredFrameQueue = new ConcurrentQueue<KeyValuePair<long, byte[]>>();
            }
        }

        private void newCameraPlay(WriteableBitmap originBitmap, WriteableBitmap transBitmap)
        {
            if (SelectedCamera != null)
            {
                capture = new VideoCapture(SelectedCamera.Index);
                capture.ImageGrabbed += ProcessFrame;
                capture.Start();

            }

        }

        private long lastTimestamp = 0;
        object obj = new object();
        private void ProcessFrame(object sender, EventArgs arg)
        {
            
            lock (obj)
            {
                if (capture != null && capture.Ptr != IntPtr.Zero && capture.Retrieve(frameBgr, 0))
                {
                    long curTimestamp = DateTime.Now.Ticks / 10000;


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
                            _originBitmap.Dispatcher?.Invoke(new Action(() =>
                            {
                                _originBitmap.WritePixels(new Int32Rect(0, 0, WIDTH, HEIGHT), frameFit.DataPointer, WIDTH * HEIGHT * 3, WIDTH * 3);
                            }), DispatcherPriority.Send);
                        }
                        catch
                        { }
                        var gap = (1000.0 / FrameRate - 1);

                        if ((curTimestamp - lastTimestamp) > gap)
                        {
                            byte[] rgbs = new byte[WIDTH * HEIGHT * 3];
                            Marshal.Copy(frameFit.DataPointer, rgbs, 0, WIDTH * HEIGHT * 3);
                            originFrameQueue.Enqueue(new KeyValuePair<long, byte[]>(curTimestamp, rgbs));
                            lastTimestamp = curTimestamp;
                        }

                    }
                }
            }
        }

        public void TranferTask()
        {
            List<byte[]> batch = new List<byte[]>();
            List<long> batchTimestap = new List<long>();
            while (IsRunning)
            {
                //if (!_isPlay) {
                //    ClearTransBuffer();
                //    _isPlay = true;
                //}
                KeyValuePair<long, byte[]> origin;
                bool dequeue = originFrameQueue.TryDequeue(out origin);
                //OriginBufferSize = originFrameQueue.Count();
                if (!dequeue || origin.Value == null)
                {
                    Thread.Sleep(30);
                    continue;
                }

                batch.Add(origin.Value);
                batchTimestap.Add(origin.Key);

                if (batch.Count >= BatchSize)
                {

                    // Transfer 
                    List<byte[]> transfferedBatch = new List<byte[]>();

                    try
                    {
                        transfferedBatch = _model.Transfer(batch);
                        for (int i = 0; i < transfferedBatch.Count; i++)
                        {
                            transferredFrameQueue.Enqueue(new KeyValuePair<long, byte[]>(batchTimestap.ElementAt(i), transfferedBatch.ElementAt(i)));
                            //TransBufferSize = transferredFrameQueue.Count();
                        }
                    }
                    catch
                    {
                        if (IsRunning)
                        {
                            handleException("Fail to transfer some frames");
                        }
                    }
                    batch.Clear();
                    batchTimestap.Clear();
                }
            }
        }

        public void TransferredPlay(WriteableBitmap transBitmap)
        {
            long lastFrameTimestamp = 0;
            long lastPlayTimestamp = 0;
            // Magic number
            int M = BatchSize;
            int N = 100;
            double a = 1.0 / ((M - N) * N);
            double b = -M * a;
            double c = 1.0;
            while (IsRunning)
            {

                KeyValuePair<long, byte[]> transferred;
                bool dequeue = transferredFrameQueue.TryDequeue(out transferred);

                //TransBufferSize = transferredFrameQueue.Count();
                if (!dequeue || transferred.Value == null)
                {
                    Thread.Sleep(30);
                    continue;
                }

                int bufferSize = transferredFrameQueue.Count();
                double alpha = (a * bufferSize * bufferSize + b * bufferSize + c);
                //long gap = (transferred.Key - lastFrameTimestamp - (DateTime.Now.Ticks / 10000 - lastPlayTimestamp)); 
                double gap = Math.Min(1000.0, alpha * (transferred.Key - lastFrameTimestamp - (DateTime.Now.Ticks / 10000 - lastPlayTimestamp)));
                
                if (lastFrameTimestamp != 0 && gap > 0)
                {
                    try
                    {
                        Thread.Sleep((int)gap);
                    }
                    catch { }
                }
                lastFrameTimestamp = transferred.Key;

                try
                {
                    transBitmap.Dispatcher?.Invoke(new Action(() =>
                    {
                        lastPlayTimestamp = DateTime.Now.Ticks / 10000;
                        transBitmap.WritePixels(new Int32Rect(0, 0, WIDTH, HEIGHT), transferred.Value, WIDTH * 3, 0);
                    }), DispatcherPriority.DataBind);
                }
                catch
                {
                }
            }
        }

        public void handleException(string errorMsg)
        {
            MessageBox.Show(errorMsg);
        }
    }
}