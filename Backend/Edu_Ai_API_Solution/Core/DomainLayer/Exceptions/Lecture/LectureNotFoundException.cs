namespace DomainLayer.Exceptions.Lecture
{
	public class LectureNotFoundException : NotFoundException
	{
		public LectureNotFoundException(int lectureId)
			: base($"Lecture with id '{lectureId}' was not found.")
		{
		}
	}
}