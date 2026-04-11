using DomainLayer.Models;

namespace ServiceLayer.Specifications.LectureSpecifications
{
	public class LecturesByCourseSpecification : BaseSpecification<Lecture, int>
	{
		public LecturesByCourseSpecification(int courseId)
			: base(l => l.CourseId == courseId)
		{
			AddInclude(l => l.CreatedBy);
			AddOrderByDescending(l => l.ScheduledAt);
		}

		public LecturesByCourseSpecification(int courseId, int lectureId)
			: base(l => l.CourseId == courseId && l.Id == lectureId)
		{
			AddInclude(l => l.CreatedBy);
		}
	}
}