using Microsoft.AspNetCore.Http;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ServiceAbstractionLayer
{
    public interface IFileStorageService
    {
    Task<string> UploadFileAsync(
             Stream fileStream,
             string fileName,
             string folder,
             string contentType);
    }
}
