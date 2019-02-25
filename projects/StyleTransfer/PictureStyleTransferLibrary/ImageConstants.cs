namespace PictureStyleTransferLibrary
{
    /// <summary>
    /// Describe the requirement on image properties.
    /// </summary>
    public class ImageConstants
    {
        /// <summary>
        /// Legal height (px) for image.
        /// </summary>
        public const int Height = 480;

        /// <summary>
        /// Legal width (px) for image.
        /// </summary>
        public const int Width = 640;

        /// <summary>
        /// Channel count (RGB) for image.
        /// </summary>
        public const int Channel = 3;

        /// <summary>
        /// Bitmap byte size for image.
        /// </summary>
        public const int Size = Height * Width * Channel;
    }
}
