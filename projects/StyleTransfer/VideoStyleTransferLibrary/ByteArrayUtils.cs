using System;
using System.Collections.Generic;
using System.Diagnostics;

namespace VideoStyleTransferLibrary
{
    class ByteArrayUtils
    {
        public static List<float> ConvertToFloatArray(List<byte[]> images)
        {
            Debug.Assert(images != null);
            var converted = new List<float>(images.Count * ImageConstants.Size);
            foreach (var frame in images)
            {
                var data = new List<float>(frame.Length);
                foreach (var item in frame) data.Add(item);
                converted.AddRange(data);
            }
            return converted;
        }

        public static List<byte[]> ConvertToByteArray(List<float> image)
        {
            Debug.Assert(image != null);
            var converted = new List<byte[]>(ImageConstants.Batch);
            for (var i = 0; i < ImageConstants.Batch; i++)
            {
                var frame = new byte[ImageConstants.Size];
                for (var j = 0; j < ImageConstants.Size; j++)
                {
                    var pos = i * ImageConstants.Size + j;
                    frame[j] = ClipPixelChannel(image[pos]);
                }
                converted.Add(frame);
            }
            return converted;
        }

        private static byte ClipPixelChannel(float val)
        {
            if (val < 0) return 0;
            if (val > 255) return 255;
            return Convert.ToByte(val);
        }
    }
}
