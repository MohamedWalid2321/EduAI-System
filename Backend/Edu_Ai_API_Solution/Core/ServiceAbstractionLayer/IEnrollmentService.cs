namespace ServiceAbstractionLayer
{
    public interface IEnrollmentService
    {
        Task AutoEnrollAsync(string studentId, CancellationToken cancellationToken = default);
        Task ReEnrollAsync(string studentId, CancellationToken cancellationToken = default);
        Task EnrollNewCourseAsync(int courseId, CancellationToken cancellationToken = default);
    }
}
