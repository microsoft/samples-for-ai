using System.Collections.Generic;

namespace VideoStyleTransferLibrary
{
    /// <summary>
    /// Provide capability of transfering image.
    /// </summary>
    public interface IModel
    {
        /// <summary>
        /// Transfer style of array-represented images by creating a new one.
        /// </summary>
        /// <param name="input">The array-represented images to transfer.</param>
        /// <returns>Array-represented images with new style.</returns>
        /// <exception cref="ArgumentNullException" />
        List<byte[]> Transfer(List<byte[]> input);
    }
}
