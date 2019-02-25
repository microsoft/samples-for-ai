using System;
using System.Collections.Generic;
using System.IO;
using Microsoft.ML.Scoring;

namespace VideoStyleTransferLibrary
{
    public class ModelTransfer : IModel
    {
        private string _modelName;

        public string ModelName
        {
            get
            {
                return _modelName;
            }
        }

        private ModelManager _manager;

        private List<long> _imageShape;

        /// <summary>
        /// Instantiate the <see cref="ModelTransfer" /> class.
        /// </summary>
        /// <param name="rootDir">Root directory containing transfer sub directories.</param>
        /// <param name="modelName">Transfer name and sub directory name containing model files.</param>
        public ModelTransfer(string rootDir, string modelName)
        {
            var path = Path.Combine(rootDir, modelName, "00000001");
            var manager = new ModelManager(path, true);
            manager.InitTensorFlowModel(modelName, Int32.MaxValue, 0, 0);

            _modelName = modelName;
            _manager = manager;
            _imageShape = new List<long> { ImageConstants.Batch, ImageConstants.Height, ImageConstants.Width, ImageConstants.Channel };
        }

        public List<byte[]> Transfer(List<byte[]> images)
        {
            var input = ByteArrayUtils.ConvertToFloatArray(images);
            var result = _manager.RunModel(
                _modelName,
                Int32.MaxValue,
                new List<string> { "input" },
                new List<Tensor> { new Tensor(input, _imageShape) },
                new List<string> { "output" });
            var transfered = new List<float>(ImageConstants.Batch * ImageConstants.Size);
            result[0].CopyTo(transfered);
            return ByteArrayUtils.ConvertToByteArray(transfered);
        }
    }
}
