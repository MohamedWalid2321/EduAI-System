<<<<<<< HEAD
﻿using Microsoft.AspNetCore.Http;
using Shared.Dtos.AssigmentDto.Request;
using Shared.Dtos.AssigmentDto.Response;
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
	public interface IAssigmentService
	{
		Task<AssigmentResponseDto> CreateOrUpdateAssigmentForCourse(int courseId, AssigmentRequestDto assigmentRequest);
		Task<IEnumerable<AssigmentResponseDto>> GetAllAssigmentsByCourseIdAsync(int courseId);
		Task<AssigmentResponseDto> GetAssigmentByIdAsync(int AssigmentId);
		Task DeleteAssigmentAsync(int AssigmentId);
		Task RemoveAttachment(Guid AttachmentId);
		Task<AssigmentResponseDto> AddAttachmentToAssigment(int AssigmentId, List<IFormFile?> Files);
	}
}
