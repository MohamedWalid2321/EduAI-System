namespace DomainLayer.Exceptions.Lecture
{
	public class LectureNotActiveException : BadRequestException
	{
		public LectureNotActiveException(int lectureId)
			: base($"Lecture with id '{lectureId}' is not currently active. The instructor has not opened this meeting yet.")
		{
		}
	}
}