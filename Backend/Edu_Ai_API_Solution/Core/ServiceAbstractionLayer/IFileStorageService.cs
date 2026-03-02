<<<<<<< HEAD
﻿using Microsoft.AspNetCore.Http;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

=======
﻿
>>>>>>> f283ebec1b7f11684dfeff6e9246326d74ada2d9
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
