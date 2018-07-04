using System.Collections.Generic;
using System.ComponentModel;
using System;
using System.IO;
using System.Net.Http;
using System.Windows.Input;
using System.Linq;
using System.Collections.ObjectModel;
using Windows.Graphics.Imaging;
using Windows.Media;
using Windows.Storage;
using Windows.UI.Xaml.Media.Imaging;
using Windows.UI.Popups;
using Windows.ApplicationModel.Resources;

namespace CustomVisionUWP.App.ViewModel
{
    public class ResultViewModel : INotifyPropertyChanged
    {
        public event PropertyChangedEventHandler PropertyChanged;
        private ResourceLoader resourceLoader;//for localization and globalization 

        public ResultViewModel()
        {
            resourceLoader = ResourceLoader.GetForCurrentView();
            //Initiate inference model
            LoadModel();
            //Binding RecognizeCommand to function EvaluateAsync
            RecognizeCommand = new RelayCommand(new Action(EvaluateAsync));

        }
        #region UI elements
        /* 
         * Overview : sources for binding
         * Data binding is a way for app's UI to display data
         * After binding, the UI element stays in sync with that data
         * 
         * In this demo program, here are five items for binding 
         * Every item is bound to a single UI element respectively
         */

        // 1. URL of bear picture 
        //    Bound to the TextBox named "InputUriBox" 
        //    Stays in sync with the text in that Textbox
        private string bearUrl = string.Empty;
        public string BearUrl
        {
            get { return bearUrl; }
            set
            {
                bearUrl = value;
                PropertyChanged?.Invoke(this, new PropertyChangedEventArgs("BearUrl"));
            }
        }

        // 2. Recognize result
        //    Bound to the ListView named "ResultArea" 
        //    contains the probability of each kind of bear
        private ObservableCollection<string> results = new ObservableCollection<string>();
        public ObservableCollection<string> Results
        {
            get { return results; }
            set
            {
                results = value;
                PropertyChanged?.Invoke(this, new PropertyChangedEventArgs("Results"));
            }
        }

        // 3. Image of bear
        //    Bound to the Image named "DisplayArea"
        //    Shows the picture get from the URL of bear picture
        private BitmapImage bearImage = null;
        public BitmapImage BearImage
        {
            get { return bearImage; }
            set
            {
                bearImage = value;
                PropertyChanged?.Invoke(this, new PropertyChangedEventArgs("BearImage"));
            }
        }

        // 4. Description of picture
        //    Bound to the TextBox named "DescribeArea"
        //    Shows the description of the Image of bear
        private string description = string.Empty;
        public string Description
        {
            get { return description; }
            set
            {
                description = value;
                PropertyChanged?.Invoke(this, new PropertyChangedEventArgs("Description"));
            }
        }
        #endregion

        // 5. Recognize Command
        //    Bound to the Click event of Button named "RecognizeButton"
        //    Bound to the function EvaluateAsync
        public ICommand RecognizeCommand { get; }

        // Model for bear classification
        private BearClassificationModel model;

        /// <summary>
        /// Load model from .onnx file in /Assets/... and create model instance
        /// </summary>
        private async void LoadModel()
        {
            StorageFile modelFile = await StorageFile.GetFileFromApplicationUriAsync(new Uri("ms-appx:///Assets/BearClassification.onnx"));
            model = await BearClassificationModel.CreateBearClassificationModel(modelFile);
        }


        /// <summary>
        /// Get picture from URL and sent it for recognition, then generate description
        /// </summary>
        private async void EvaluateAsync()
        {
            try
            {
                /*
                 * Overview : get picture, sent it to model and generate description 
                 * The model can infer one image at once
                 * The input of the model is a VideoFrame at any size
                 * The output of the model is a dictionary, the key is name of bear and the value is the probability
                 * 
                 * In this demo program, we get picture from the URL
                 * So, before the recognition, we need to get picture from URL and covert it to VideoFrame
                 * Then, we sent it for recognition and generate description
                 */

                // 1. get picture from URL and update BearImage
                BearImage = new BitmapImage(new Uri(BearUrl, UriKind.Absolute));

                // 2. create VideoFrame from URL by read data as a stream
                var response = await new HttpClient().GetAsync(BearUrl);
                var stream = await response.Content.ReadAsStreamAsync();
                BitmapDecoder decoder = await BitmapDecoder.CreateAsync(stream.AsRandomAccessStream());
                VideoFrame imageFrame = VideoFrame.CreateWithSoftwareBitmap(await decoder.GetSoftwareBitmapAsync());

                // 3. sent VideoFrame to recognition and update Results
                var result = await model.EvaluateAsync(new BearClassificationModelInput() { data = imageFrame });
                var resultDescend = result.loss.OrderByDescending(p => p.Value).ToDictionary(p => p.Key, o => o.Value);

                // 4. generate description of the picture
                Description = DescribResult(resultDescend.First().Key, resultDescend.First().Value);

                Results.Clear();
                foreach (KeyValuePair<string, float> kvp in resultDescend)
                {
                    Results.Add(resourceLoader.GetString(kvp.Key) + " : " + kvp.Value.ToString("0.000"));
                }
            }
            catch (Exception ex)
            {
                MessageDialog a = new MessageDialog(String.Format(resourceLoader.GetString("ERROR_MESSAGE"), ex.Message));
                await a.ShowAsync();
            }
        }


        /// <summary>
        /// Generate description of the picture according to the kind of bear and probability
        /// </summary>
        /// <param name="name">name of the bear</param>
        /// <param name="weight">probability of the bear</param>
        /// <returns></returns>
        public string DescribResult(string name, double weight)
        {
            string confidence = string.Empty;
            if (weight < 0.5)
            {
                confidence = "might";
            }
            else if (weight >= 0.5 && weight < 0.8)
            {
                confidence = "may";
            }
            else
            {
                confidence = "must";
            }
            if (name == "NONE")
            {
                return String.Format(resourceLoader.GetString("TEXT_NEGATIVE"), resourceLoader.GetString(confidence));
            }
            else
            {
                string translatedName = resourceLoader.GetString(name);
                if (string.IsNullOrEmpty(translatedName))
                {
                    translatedName = name;
                }
                return String.Format(resourceLoader.GetString("TEXT_POSITIVE"), resourceLoader.GetString(confidence), translatedName);
            }
        }
    }
}
