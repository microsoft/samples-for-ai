using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Windows.Media;
using Windows.Storage;
using Windows.AI.MachineLearning.Preview;


/// <summary>
/// These three class is generated automatically
/// </summary>
namespace CustomVisionUWP.App
{
    /// <summary>
    /// Input form of BearClassificationModel
    /// </summary>
    public sealed class BearClassificationModelInput
    {
        public VideoFrame data { get; set; }
    }

    /// <summary>
    /// Output form of BearClassificationModel
    /// Contains a list and a dictionary
    /// The list contains labels of bears in descending order according to the probability
    /// The dictionary contains each label and its probability
    /// </summary>
    public sealed class BearClassificationModelOutput
    {
        public IList<string> classLabel { get; set; }
        public IDictionary<string, float> loss { get; set; }
        public BearClassificationModelOutput()
        {
            this.classLabel = new List<string>();
            this.loss = new Dictionary<string, float>()
            {
                { "Black Bear", float.NaN },
                { "Brown Bear", float.NaN },
                { "NONE", float.NaN },
                { "Panda", float.NaN },
                { "Polar Bear", float.NaN },
            };
        }
    }

    /// <summary>
    /// There are two function in this model
    /// CreateBearClassificationModel is for create model, returns instance of BearClassification model from model path
    /// EvaluateAsync is for evaluation, returns the results of evaluation in the form of BearClassificationModelOutput
    /// </summary>
    public sealed class BearClassificationModel
    {
        private LearningModelPreview learningModel;
        /// <summary>
        /// Create model, returns instance of BearClassification model from model path
        /// </summary>
        /// <param name="file">The .onnx file location</param>
        /// <returns></returns>
        public static async Task<BearClassificationModel> CreateBearClassificationModel(StorageFile file)
        {
            LearningModelPreview learningModel = await LearningModelPreview.LoadModelFromStorageFileAsync(file);
            BearClassificationModel model = new BearClassificationModel();
            model.learningModel = learningModel;
            return model;
        }
        /// <summary>
        /// Evaluation, returns the results in the form of BearClassificationModelOutput
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        public async Task<BearClassificationModelOutput> EvaluateAsync(BearClassificationModelInput input)
        {
            BearClassificationModelOutput output = new BearClassificationModelOutput();
            LearningModelBindingPreview binding = new LearningModelBindingPreview(learningModel);
            binding.Bind("data", input.data);
            binding.Bind("classLabel", output.classLabel);
            binding.Bind("loss", output.loss);
            LearningModelEvaluationResultPreview evalResult = await learningModel.EvaluateAsync(binding, string.Empty);
            return output;
        }
    }
}
