using Shared.Dtos.LectureDto.Request;
using System.Threading;

namespace PresentationLayer.Controllers
{
	public class LectureController(IServiceManager serviceManager) : ApiControllerBase
	{
		// GET /api/lecture/course/{courseId}
		[HttpGet("course/{courseId}")]
		[HasPermission(Permissions.GetCourse)]
		public async Task<IActionResult> GetAllByCourse(int courseId, CancellationToken cancellationToken)
		{
			var lectures = await serviceManager.LectureService.GetAllByCourseAsync(courseId, cancellationToken);
			return Ok(lectures);
		}

		// GET /api/lecture/{lectureId}/course/{courseId}
		[HttpGet("{lectureId}/course/{courseId}")]
		[HasPermission(Permissions.GetCourse)]
		public async Task<IActionResult> GetById(int courseId, int lectureId, CancellationToken cancellationToken)
		{
			var lecture = await serviceManager.LectureService.GetByIdAsync(courseId, lectureId, cancellationToken);
			return Ok(lecture);
		}

		// POST /api/lecture/course/{courseId}
		[HttpPost("course/{courseId}")]
		[HasPermission(Permissions.CreateLecture)]
		public async Task<IActionResult> Create(int courseId, [FromBody] CreateLectureRequest request, CancellationToken cancellationToken)
		{
			var createdById = User.GetUserId()!;
			var result = await serviceManager.LectureService.CreateAsync(courseId, createdById, request, cancellationToken);
			return CreatedAtAction(nameof(GetById), new { courseId, lectureId = result.Id }, result);
		}

		// PUT /api/lecture/{lectureId}/course/{courseId}
		[HttpPut("{lectureId}/course/{courseId}")]
		[HasPermission(Permissions.UpdateLecture)]
		public async Task<IActionResult> Update(int courseId, int lectureId, [FromBody] UpdateLectureRequest request, CancellationToken cancellationToken)
		{
			await serviceManager.LectureService.UpdateAsync(courseId, lectureId, request, cancellationToken);
			return NoContent();
		}

		// DELETE /api/lecture/{lectureId}/course/{courseId}
		[HttpDelete("{lectureId}/course/{courseId}")]
		[HasPermission(Permissions.DeleteLecture)]
		public async Task<IActionResult> Delete(int courseId, int lectureId, CancellationToken cancellationToken)
		{
			await serviceManager.LectureService.DeleteAsync(courseId, lectureId, cancellationToken);
			return NoContent();
		}

		// PUT /api/lecture/{lectureId}/course/{courseId}/toggle-active
		[HttpPut("{lectureId}/course/{courseId}/toggle-active")]
		[HasPermission(Permissions.UpdateLecture)]
		public async Task<IActionResult> ToggleActive(int courseId, int lectureId, CancellationToken cancellationToken)
		{
			await serviceManager.LectureService.ToggleActiveAsync(courseId, lectureId, cancellationToken);
			return NoContent();
		}

		// GET /api/lecture/{lectureId}/join  — student joins a live meeting
		[HttpGet("{lectureId}/join")]
		[HasPermission(Permissions.JoinLecture)]
		public async Task<IActionResult> Join(int lectureId, CancellationToken cancellationToken)
		{
			var userId = User.GetUserId()!;
			var result = await serviceManager.LectureService.JoinAsync(lectureId, userId, cancellationToken);
			return Ok(result);
		}
	}
}