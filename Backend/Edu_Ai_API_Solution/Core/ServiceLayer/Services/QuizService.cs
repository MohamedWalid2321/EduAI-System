using ServiceLayer.Specifications.AttemptedQuizSpecification;
using Shared.Dtos.AttemptQuiz.Response;

namespace ServiceLayer.Services
{
	public class QuizService(IUnitOfWork unitOfWork) : IQuizService
	{
		private readonly IUnitOfWork _unitOfWork = unitOfWork;

public async Task<QuizResponseDto> CreateOrUpdateQuizAsync(int CourseId, QuizRequestDto quizRequest, CancellationToken cancellationToken = default)
        {
            var quizRepository = _unitOfWork.GetRepository<Quiz, int>();
            var courseRepository = _unitOfWork.GetRepository<Course, int>();
			

            // Verify course exists
            var course = await courseRepository.GetByIdAsync(CourseId, cancellationToken);
            if (course is null)
            {
                throw new CourseNotFoundException(CourseId);
            }

            if (quizRequest.Id.HasValue && quizRequest.Id > 0)
            {
                // Update
                var existingQuiz = await quizRepository.GetByIdAsync(quizRequest.Id.Value, cancellationToken);
                if (existingQuiz == null)
                {
                    throw new QuizNotFoundException(quizRequest.Id.Value);
                }

                // Update properties on tracked entity
                existingQuiz.Title = quizRequest.Title;
                existingQuiz.Description = quizRequest.Description;
                existingQuiz.ScheduledDate = quizRequest.ScheduledDate;
                existingQuiz.Duration = quizRequest.Duration;
                existingQuiz.TotalMarks = quizRequest.TotalMarks;
                existingQuiz.CourseId = CourseId;
				existingQuiz.IsActive = quizRequest.IsActive; // Ensure quiz is active when updated
				existingQuiz.QuizCode = Guid.NewGuid()
										.ToString("N")   
										.Substring(0, 8)
										.ToUpper();


                quizRepository.Update(existingQuiz);
            }
            else
            {
				
                // Create 
                var quizEntity = quizRequest.Adapt<Quiz>();
				quizEntity.QuizCode = Guid.NewGuid()
                                            .ToString("N")
                                            .Substring(0, 8)
                                            .ToUpper();
                quizEntity.CourseId = CourseId;

                await quizRepository.AddAsync(quizEntity, cancellationToken);
            }

            await _unitOfWork.SaveChangesAsync(cancellationToken);

            // Reload with includes for the response
            var quizSpec = new QuizSpecification(quizRequest.Id.HasValue && quizRequest.Id > 0 ? quizRequest.Id.Value : course.Quizzes?.LastOrDefault()?.Id ?? 0);
            var updatedQuiz = await quizRepository.GetByIdAsync(quizSpec, cancellationToken);
            return updatedQuiz!.Adapt<QuizResponseDto>();
        }

		public async Task DeleteQuizAsync(int quizId, CancellationToken cancellationToken = default)
		{
			var quizRepository = _unitOfWork.GetRepository<Quiz, int>();
			var quizEntity = await quizRepository.GetByIdAsync(quizId, cancellationToken);
			if (quizEntity is null)
			{
				throw new QuizNotFoundException(quizId);
			}
			quizRepository.Delete(quizEntity);
			await _unitOfWork.SaveChangesAsync(cancellationToken);
		}

		public async Task<IEnumerable<QuizResponseDto>> GetAllQuizzesAsync(CancellationToken cancellationToken = default)
		{
			var quizRepository = _unitOfWork.GetRepository<Quiz, int>();
			var quizSpec = new QuizSpecification();
			var quizEntities = await quizRepository.GetAllAsync(quizSpec, cancellationToken);
            return quizEntities.Adapt<IEnumerable<QuizResponseDto>>();
		}

		public async Task<IEnumerable<QuizResponseDto>> GetAllQuizzesForCourse(int CourseId, bool onlyActive, CancellationToken cancellationToken = default)
		{
			var quizRepository = _unitOfWork.GetRepository<Quiz, int>();
			var quizSpec = onlyActive
				? new QuizByCourseIdSpecification(CourseId, onlyActive: true)
				: new QuizByCourseIdSpecification(CourseId);
			var quizEntities = await quizRepository.GetAllAsync(quizSpec, cancellationToken);
			if (quizEntities is null || !quizEntities.Any())
			{
				throw new QuizzesInCourseNotFoundException(CourseId);
            }
            return quizEntities.Adapt<IEnumerable<QuizResponseDto>>();
		}
		public async Task<IEnumerable<QuizForStudentResponseDto>> GetAllQuizzesForCourseAsStudentAsync(
			int courseId,
			string studentId,
			CancellationToken cancellationToken = default)
		{
			// 1. Fetch active quizzes for the course
			var quizRepository    = _unitOfWork.GetRepository<Quiz, int>();
			var attemptRepository = _unitOfWork.GetRepository<QuizAttempt, int>();

			var quizSpec    = new QuizByCourseIdSpecification(courseId, onlyActive: true);
			var quizEntities = await quizRepository.GetAllAsync(quizSpec, cancellationToken);

			if (quizEntities is null || !quizEntities.Any())
				throw new QuizzesInCourseNotFoundException(courseId);

			// 2. Fetch all submitted attempts for this student in this course
			var attemptSpec = new StudentQuizAttemptsByCourseSpecification(courseId, studentId);
			var attempts    = await attemptRepository.GetAllAsync(attemptSpec, cancellationToken);

			// Key: QuizId → attempt (there should be at most one submitted attempt per quiz)
			var attemptByQuizId = attempts.ToDictionary(a => a.QuizId);

			// 3. Merge
			var result = quizEntities.Select(q =>
			{
				var dto = q.Adapt<QuizForStudentResponseDto>();

				if (attemptByQuizId.TryGetValue(q.Id, out var attempt))
				{
					dto.IsSubmitted = true;
					dto.Score       = attempt.Score;
					dto.SubmittedAt = attempt.SubmittedAt;
				}

				return dto;
			});

			return result;
		}

		public async Task<QuizResponseInDetailsDto> GetQuizByIdAsync(int quizId, CancellationToken cancellationToken = default)
		{
			var quizRepository = _unitOfWork.GetRepository<Quiz, int>();
			var quizSpec = new QuizSpecification(quizId);
			var quizEntity = await quizRepository.GetByIdAsync(quizSpec, cancellationToken);
			if (quizEntity is null)
			{
				throw new QuizNotFoundException(quizId);
			}
			return quizEntity.Adapt<QuizResponseInDetailsDto>();
		}

		public async Task<bool> ToggleQuizActiveAsync(int quizId, CancellationToken cancellationToken = default)
		{
			var quizRepository = _unitOfWork.GetRepository<Quiz, int>();
			var quizEntity = await quizRepository.GetByIdAsync(quizId, cancellationToken);
			if (quizEntity is null)
			{
				throw new QuizNotFoundException(quizId);
			}

			quizEntity.IsActive = !quizEntity.IsActive;
			quizRepository.Update(quizEntity);
			await _unitOfWork.SaveChangesAsync(cancellationToken);

			return quizEntity.IsActive; // returns the new state
		}
	}
}
