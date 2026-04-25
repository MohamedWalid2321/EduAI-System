using Shared.Dtos.LectureDto.Request;

namespace PresentationLayer.Controllers
{
	public class LectureController(IServiceManager serviceManager) : ApiControllerBase
	{
		// GET /api/lecture/course/{courseId}
		[HttpGet("course/{courseId}")]
		[HasPermission(Permissions.GetCourse)]
		public async Task<IActionResult> GetAllByCourse(int courseId)
		{
			var lectures = await serviceManager.LectureService.GetAllByCourseAsync(courseId);
			return Ok(lectures);
		}

		// GET /api/lecture/{lectureId}/course/{courseId}
		[HttpGet("{lectureId}/course/{courseId}")]
		[HasPermission(Permissions.GetCourse)]
		public async Task<IActionResult> GetById(int courseId, int lectureId)
		{
			var lecture = await serviceManager.LectureService.GetByIdAsync(courseId, lectureId);
			return Ok(lecture);
		}

		// POST /api/lecture/course/{courseId}
		[HttpPost("course/{courseId}")]
		[HasPermission(Permissions.CreateLecture)]
		public async Task<IActionResult> Create(int courseId, [FromBody] CreateLectureRequest request)
		{
			var createdById = User.GetUserId()!;
			var result = await serviceManager.LectureService.CreateAsync(courseId, createdById, request);
			return CreatedAtAction(nameof(GetById), new { courseId, lectureId = result.Id }, result);
		}

		// PUT /api/lecture/{lectureId}/course/{courseId}
		[HttpPut("{lectureId}/course/{courseId}")]
		[HasPermission(Permissions.UpdateLecture)]
		public async Task<IActionResult> Update(int courseId, int lectureId, [FromBody] UpdateLectureRequest request)
		{
			await serviceManager.LectureService.UpdateAsync(courseId, lectureId, request);
			return NoContent();
		}

		// DELETE /api/lecture/{lectureId}/course/{courseId}
		[HttpDelete("{lectureId}/course/{courseId}")]
		[HasPermission(Permissions.DeleteLecture)]
		public async Task<IActionResult> Delete(int courseId, int lectureId)
		{
			await serviceManager.LectureService.DeleteAsync(courseId, lectureId);
			return NoContent();
		}

		// PUT /api/lecture/{lectureId}/course/{courseId}/toggle-active
		[HttpPut("{lectureId}/course/{courseId}/toggle-active")]
		[HasPermission(Permissions.UpdateLecture)]
		public async Task<IActionResult> ToggleActive(int courseId, int lectureId)
		{
			await serviceManager.LectureService.ToggleActiveAsync(courseId, lectureId);
			return NoContent();
		}

		// GET /api/lecture/{lectureId}/join  — student joins a live meeting
		[HttpGet("{lectureId}/join")]
		[HasPermission(Permissions.JoinLecture)]
		public async Task<IActionResult> Join(int lectureId)
		{
			var userId = User.GetUserId()!;
			var result = await serviceManager.LectureService.JoinAsync(lectureId, userId);
			return Ok(result);
		}
	}
}