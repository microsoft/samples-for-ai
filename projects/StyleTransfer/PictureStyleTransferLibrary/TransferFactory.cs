using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Threading.Tasks;

namespace PictureStyleTransferLibrary
{
    /// <summary>
    /// Maintain a list of transfer models.
    /// </summary>
    public class TransferFactory
    {
        private Dictionary<string, IModel> _modelMap;

        private object _lock;

        private Task[] _initTasks;

        /// <summary>
        /// Instantiate the <see cref="TransferFactory" /> class.
        /// </summary>
        /// <param name="syncInit">Initialize transfers one by one in a blocking way.</param>
        public TransferFactory(bool syncInit = false)
        {
            var dllPath = Assembly.GetExecutingAssembly().Location;
            var rootDir = Path.GetDirectoryName(dllPath);
            var modelDir = Path.Combine(rootDir, "models");
            var models = Directory.GetDirectories(modelDir);

            _modelMap = new Dictionary<string, IModel>(models.Length);
            _initTasks = new Task[models.Length];
            _lock = new object();
            for (var i = 0; i < models.Length; i++)
            {
                var modelName = Path.GetFileName(models[i]);
                var task = new Task(() =>
                {
                    var transfer = new ModelTransfer(modelDir, modelName);
                    lock (_lock)
                    {
                        _modelMap.Add(modelName, transfer);
                    }
                });
                task.Start();
                _initTasks[i] = task;
            }
            if (syncInit) WaitInitialization();
        }

        /// <summary>
        /// Return all available transfers.
        /// </summary>
        public string[] Supported
        {
            get
            {
                WaitInitialization();
                var ret = _modelMap.Keys.ToArray();
                if (ret != null)
                {
                    Array.Sort(ret);
                }

                return ret;
            }
        }

        /// <summary>
        /// Fetch the transfer model by name.
        /// </summary>
        /// <param name="name">Transfer name.</param>
        /// <returns>A <see cref="IModel" /> instance for serving.</returns>
        public IModel GetTransfer(string name)
        {
            if (string.IsNullOrWhiteSpace(name)) throw new ArgumentNullException("name");
            WaitInitialization();
            return _modelMap.ContainsKey(name) ? _modelMap[name] : null;
        }

        private void WaitInitialization()
        {
            if (_initTasks != null)
            {
                lock (this)
                {
                    if (_initTasks != null)
                    {
                        Task.WaitAll(_initTasks);
                        _initTasks = null;
                    }
                }
            }
        }
    }
}
