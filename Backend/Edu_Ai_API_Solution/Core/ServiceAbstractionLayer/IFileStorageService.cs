namespace ServiceAbstractionLayer
{
    public interface IFileStorageService
    {
        Task<string> UploadFileAsync(
            Stream fileStream,
            string fileName,
            string folder,
            string contentType);

        Task DeleteFileAsync(string fileUrl);
    }
}
