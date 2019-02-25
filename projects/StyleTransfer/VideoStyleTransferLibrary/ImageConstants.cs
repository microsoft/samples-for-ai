using System.IO;
using System.Reflection;

namespace VideoStyleTransferLibrary
{
    /// <summary>
    /// Describe the requirement on image properties.
    /// </summary>
    public class ImageConstants
    {
        static ImageConstants()
        {
            var dllPath = Assembly.GetExecutingAssembly().Location;
            var rootDir = Path.GetDirectoryName(dllPath);
            var iniPath = Path.Combine(rootDir, "ImageConstants.ini");
            foreach (var line in File.ReadLines(iniPath))
            {
                var config = line.ToLower();
                if (config.StartsWith("width="))
                {
                    Width = int.Parse(config.Substring(6));
                }
                else if (config.StartsWith("height="))
                {
                    Height = int.Parse(config.Substring(7));
                }
                else if (config.StartsWith("channel="))
                {
                    Channel = int.Parse(config.Substring(8));
                }
                else if (config.StartsWith("batch="))
                {
                    Batch = int.Parse(config.Substring(6));
                }
            }
        }

        /// <summary>
        /// Legal height (px) for image.
        /// </summary>
        public static int Height = 240;

        /// <summary>
        /// Legal width (px) for image.
        /// </summary>
        public static int Width = 320;

        /// <summary>
        /// Channel count (RGB) for image.
        /// </summary>
        public static int Channel = 3;

        /// <summary>
        /// Bitmap byte size for image.
        /// </summary>
        public static int Size = Height * Width * Channel;

        /// <summary>
        /// Batch size for transferring images in once.
        /// </summary>
        public static int Batch = 16;
    }
}
