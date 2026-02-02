using Microsoft.AspNetCore.Http;
using Shared.Dtos.CourseDto.Request;
using Shared.Dtos.CourseDto.Response;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ServiceAbstractionLayer
{
	public interface ICourseService
	{
		Task<CourseResponseDto> CreateOrUpdateCourseAsync(CourseRequestDto courseDto , IFormFile? ImageFile);
		Task<IEnumerable<CourseResponseDto>> GetAllCourseAsync();
		Task<CourseResponseDto> GetCourseByIdAsync(int courseId);
		Task DeleteCourseAsync(int courseId);


	}
}
