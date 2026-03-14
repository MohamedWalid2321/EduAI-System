namespace ServiceAbstractionLayer
{
    public interface IEnrollmentService
    {
        Task AutoEnrollAsync(string studentId);
        Task ReEnrollAsync(string studentId);
        Task EnrollNewCourseAsync(int courseId);
    }
}
