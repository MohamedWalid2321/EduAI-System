namespace ServiceLayer.Services
{
    // This service is responsible for uploading files to Bunny.net
    public class BunnyNetService : IFileStorageService
    {
        // HttpClient is used to send HTTP requests (PUT, GET)
        // It is injected via Dependency Injection to:
        // 1) Reuse connections
        // 2) Avoid socket exhaustion
        private readonly HttpClient _httpClient;

        // Base CDN URL used to generate the public file URL after upload
        private readonly string _cdnBaseUrl;

        // Constructor receives dependencies from the DI container
        public BunnyNetService(HttpClient httpClient, IConfiguration configuration)
        {
            // Assign injected HttpClient
            _httpClient = httpClient;

            // Read the "Bunny" section from appsettings.json
            var bunny = configuration.GetSection("Bunny");

            // StorageUrl is the Bunny Storage API base URL
            var storageUrl = bunny["StorageUrl"]
                ?? throw new InvalidOperationException("Bunny:StorageUrl not configured");

            // ApiKey is required for authentication with Bunny Storage
            var apiKey = bunny["ApiKey"]
                ?? throw new InvalidOperationException("Bunny:ApiKey not configured");

            // CDN Base URL used to build the final public file URL
            _cdnBaseUrl = bunny["CdnBaseUrl"]
                ?? throw new InvalidOperationException("Bunny:CdnBaseUrl not configured");

            // Configure HttpClient base address
            // TrimEnd avoids double slashes in URLs
            _httpClient.BaseAddress = new Uri(storageUrl.TrimEnd('/') + "/");

            // Add Bunny authentication header if not already added
            // Prevents adding duplicate headers
            if (!_httpClient.DefaultRequestHeaders.Contains("AccessKey"))
            {
                _httpClient.DefaultRequestHeaders.Add("AccessKey", apiKey);
            }
        }

        // Uploads a file stream to Bunny Storage
        // Returns the public CDN URL of the uploaded file
        public async Task<string> UploadFileAsync(
            Stream fileStream,
            string fileName,
            string folder,
            string contentType)
        {
            // Validate file stream
            // Prevents null reference errors
            if (fileStream == null)
                throw new ArgumentNullException(nameof(fileStream));

            // Validate file name
            if (string.IsNullOrWhiteSpace(fileName))
                throw new ArgumentException("File name cannot be empty", nameof(fileName));

            // Validate content type
            if (string.IsNullOrWhiteSpace(contentType))
                throw new ArgumentException("Content type cannot be empty", nameof(contentType));

            // Clean folder path by removing leading/trailing slashes
            // This avoids malformed URLs
            var cleanFolder = (folder ?? string.Empty).Trim('/');

            // Build final storage path
            // If folder is empty → upload file to root
            var rawPath = string.IsNullOrEmpty(cleanFolder)
                ? fileName.TrimStart('/')
                : $"{cleanFolder}/{fileName.TrimStart('/')}";

            // URL-encode each path segment so spaces, @, and other special
            // characters are transmitted correctly to Bunny Storage
            var path = string.Join("/",
                rawPath.Split('/').Select(Uri.EscapeDataString));

            // Ensure stream starts from the beginning
            // Important if the stream was read before uploading
            if (fileStream.CanSeek && fileStream.Position != 0)
            {
                fileStream.Position = 0;
            }

            // Wrap the stream inside StreamContent for HTTP upload
            using var content = new StreamContent(fileStream);

            // Set content type (image/png, application/pdf, video/mp4)
            content.Headers.ContentType = new MediaTypeHeaderValue(contentType);

            // Send PUT request to Bunny Storage
            var response = await _httpClient.PutAsync(path, content);

            // If upload fails, read the error response
            // and throw a meaningful exception
            if (!response.IsSuccessStatusCode)
            {
                var error = await response.Content.ReadAsStringAsync();
                throw new HttpRequestException(
                    $"Bunny.net Error: {response.StatusCode} - {error}");
            }

            // Return the public CDN URL of the uploaded file
            return $"{_cdnBaseUrl.TrimEnd('/')}/{path}";
        }

        // Deletes a file from Bunny Storage using its CDN URL
        public async Task DeleteFileAsync(string fileUrl)
        {
            // Validate file URL
            if (string.IsNullOrWhiteSpace(fileUrl))
                throw new ArgumentException("File URL cannot be empty", nameof(fileUrl));

            // Extract the relative path from the full CDN URL
            // Example: https://lms-videos-h.b-cdn.net/Courses/images/photo.png
            // Becomes: Courses/images/photo.png
            var cdnBaseUrlTrimmed = _cdnBaseUrl.TrimEnd('/');
            
            if (!fileUrl.StartsWith(cdnBaseUrlTrimmed, StringComparison.OrdinalIgnoreCase))
                throw new ArgumentException("File URL does not belong to the configured CDN", nameof(fileUrl));

            var relativePath = fileUrl
                .Substring(cdnBaseUrlTrimmed.Length)
                .TrimStart('/');

            if (string.IsNullOrWhiteSpace(relativePath))
                throw new ArgumentException("Could not extract file path from URL", nameof(fileUrl));

            // Normalize then encode each segment: first unescape to handle URLs
            // already stored encoded in the DB (prevents double-encoding),
            // then re-encode so raw characters (@, spaces) are always transmitted
            // correctly to Bunny Storage
            var encodedPath = string.Join("/",
                relativePath.Split('/').Select(s => Uri.EscapeDataString(Uri.UnescapeDataString(s))));

            // Send DELETE request to Bunny Storage
            var response = await _httpClient.DeleteAsync(encodedPath);

            // If deletion fails, read the error response
            // Note: 404 is acceptable (file already deleted)
            if (!response.IsSuccessStatusCode && response.StatusCode != System.Net.HttpStatusCode.NotFound)
            {
                var error = await response.Content.ReadAsStringAsync();
                throw new HttpRequestException(
                    $"Bunny.net Delete Error: {response.StatusCode} - {error}");
            }
        }
    }
}