using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;

namespace PictureStyleTransferLibrary
{
    class ByteArrayUtils
    {
        public static List<float> ConvertToFloatArray(byte[] image)
        {
            Debug.Assert(image != null);
            Debug.Assert(image.Length == ImageConstants.Size);
            var data = new List<float>(image.Length);
            foreach (var item in image) data.Add(item);
            return data;
        }

        public static byte[] ConvertToByteArray(List<float> image)
        {
            Debug.Assert(image != null);
            return image.Select(item => ClipPixelChannel(item)).ToArray();
        }

        private static byte ClipPixelChannel(float val)
        {
            if (val < 0) return 0;
            if (val > 255) return 255;
            return Convert.ToByte(val);
        }
    }
}
