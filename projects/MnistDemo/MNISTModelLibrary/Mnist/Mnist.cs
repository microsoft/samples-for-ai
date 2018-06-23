using Microsoft.ML.Scoring;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;

namespace MNISTModelLibrary
{
    public partial class Mnist
    {
        const string modelName = "Mnist";
        private ModelManager manager;

        private static List<string> inferInputNames = new List<string> { "inputs" };
        private static List<string> inferOutputNames = new List<string> { "outputs" };

        /// <summary>
        /// Returns an instance of Mnist model.
        /// </summary>
        public Mnist()
        {
            string codeBase = Assembly.GetExecutingAssembly().CodeBase;
            UriBuilder uri = new UriBuilder(codeBase);
            string dllpath = Uri.UnescapeDataString(uri.Path);
            string modelpath = Path.Combine(Path.GetDirectoryName(dllpath), "Mnist");
            string path = Path.Combine(modelpath, "00000001");
            manager = new ModelManager(path, true);
            manager.InitModel(modelName, int.MaxValue);
        }

        /// <summary>
        /// Returns instance of Mnist model instantiated from exported model path.
        /// </summary>
        /// <param name="path">Exported model directory.</param>
        public Mnist(string path)
        {
            manager = new ModelManager(path, true);
            manager.InitModel(modelName, int.MaxValue);
        }

        /// <summary>
        /// Runs inference on Mnist model for a batch of inputs.
        /// The shape of each input is the same as that for the non-batch case above.
        /// </summary>
        public IEnumerable<IEnumerable<long>> Infer(IEnumerable<IEnumerable<float>> inputsBatch)
        {
            List<float> inputsCombined = new List<float>();
            foreach (var input in inputsBatch)
            {
                inputsCombined.AddRange(input);
            }

            List<Tensor> result = manager.RunModel(
                modelName,
                int.MaxValue,
                inferInputNames,
                new List<Tensor> { new Tensor(inputsCombined, new List<long> { inputsBatch.LongCount(), 28, 28, 1 }) },
                inferOutputNames
            );

            int outputsBatchNum = (int)result[0].GetShape()[0];
            int outputsBatchSize = (int)result[0].GetShape().Aggregate((a, x) => a * x) / outputsBatchNum;
            for (int batchNum = 0, offset = 0; batchNum < outputsBatchNum; batchNum++, offset += outputsBatchSize)
            {
                List<long> tmp = new List<long>();
                result[0].CopyTo(tmp, offset, outputsBatchSize);
                yield return tmp;
            }
        }
    } // END OF CLASS
} // END OF NAMESPACE
