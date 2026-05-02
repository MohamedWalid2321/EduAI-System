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

		public async Task<IEnumerable<QuizResponseDto>> GetAllQuizzesForCourse(int CourseId, CancellationToken cancellationToken = default)
		{
			var quizRepository = _unitOfWork.GetRepository<Quiz, int>();
			var quizSpec = new QuizByCourseIdSpecification(CourseId);
			var quizEntities = await quizRepository.GetAllAsync(quizSpec, cancellationToken);
			if (quizEntities is null || !quizEntities.Any())
			{
				throw new QuizzesInCourseNotFoundException(CourseId);
            }
            return quizEntities.Adapt<IEnumerable<QuizResponseDto>>();
		}

		public async Task<QuizResponseDto> GetQuizByIdAsync(int quizId, CancellationToken cancellationToken = default)
		{
			var quizRepository = _unitOfWork.GetRepository<Quiz, int>();
			var quizSpec = new QuizSpecification(quizId);
			var quizEntity = await quizRepository.GetByIdAsync(quizSpec, cancellationToken);
			if (quizEntity is null)
			{
				throw new QuizNotFoundException(quizId);
			}
			return quizEntity.Adapt<QuizResponseDto>();
		}
	}
}
