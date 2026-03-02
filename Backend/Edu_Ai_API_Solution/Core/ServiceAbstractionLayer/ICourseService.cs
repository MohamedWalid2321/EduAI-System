<<<<<<< HEAD
﻿using Microsoft.AspNetCore.Http;
using Shared.Dtos.CourseDto.Request;
using Shared.Dtos.CourseDto.Response;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ServiceAbstractionLayer
=======
﻿namespace ServiceAbstractionLayer
>>>>>>> f283ebec1b7f11684dfeff6e9246326d74ada2d9
{
	public interface ICourseService
	{
		Task<CourseResponseDto> CreateOrUpdateCourseAsync(CourseRequestDto courseDto , IFormFile? ImageFile);
		Task<IEnumerable<CourseResponseDto>> GetAllCourseAsync();
		Task<CourseResponseDto> GetCourseByIdAsync(int courseId);
		Task DeleteCourseAsync(int courseId);


	}
}
