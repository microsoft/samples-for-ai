namespace PictureStyleTransferLibrary
{
    /// <summary>
    /// Provide capability of transfering image.
    /// </summary>
    public interface IModel
    {
        /// <summary>
        /// Transfer style of a array-represented image by creating a new one.
        /// </summary>
        /// <param name="input">The array-represented image to transfer.</param>
        /// <returns>An array-represented image with new style.</returns>
        /// <exception cref="ArgumentNullException" />
        byte[] Transfer(byte[] input);
    }
}
