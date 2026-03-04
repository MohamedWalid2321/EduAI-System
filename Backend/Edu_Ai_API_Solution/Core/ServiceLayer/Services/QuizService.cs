namespace ServiceLayer.Services
{
	public class QuizService(IUnitOfWork unitOfWork) : IQuizService
	{
		private readonly IUnitOfWork _unitOfWork = unitOfWork;

        public async Task<QuizResponseDto> CreateOrUpdateQuizAsync(int CourseId, QuizRequestDto quizRequest)
        {
            var quizRepository = _unitOfWork.GetRepository<Quiz, int>();
            var courseRepository = _unitOfWork.GetRepository<Course, int>();
			

            // Verify course exists
            var course = await courseRepository.GetByIdAsync(CourseId);
            if (course is null)
            {
                throw new CourseNotFoundException(CourseId);
            }

            if (quizRequest.Id.HasValue && quizRequest.Id > 0)
            {
                // Update
                var existingQuiz = await quizRepository.GetByIdAsync(quizRequest.Id.Value);
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

                await quizRepository.AddAsync(quizEntity);
            }

            await _unitOfWork.SaveChangesAsync();

            // Reload with includes for the response
            var quizSpec = new QuizSpecification(quizRequest.Id.HasValue && quizRequest.Id > 0 ? quizRequest.Id.Value : course.Quizzes?.LastOrDefault()?.Id ?? 0);
            var updatedQuiz = await quizRepository.GetByIdAsync(quizSpec);
            return updatedQuiz!.Adapt<QuizResponseDto>();
        }

     /*   public async Task<QuizResponseDto> AddQuestionToQuiz(int QuizId, ICollection<QuestionRequestDto> Questions)
		{
			var quizRepository = _unitOfWork.GetRepository<Quiz, int>();
			var questionRepository = _unitOfWork.GetRepository<QuizQuestion, int>();
			var choiceRepository = _unitOfWork.GetRepository<QuestionChoices, int>();

			// Verify quiz exists
			var quizEntity = await quizRepository.GetByIdAsync(QuizId);
			if (quizEntity is null)
			{
				throw new QuizNotFoundException(QuizId);
			}

			foreach (var questionDto in Questions)
			{
				var questionEntity = new QuizQuestion
				{
					QuestionText = questionDto.QuestionText,
					QuestionType = Enum.Parse<DomainLayer.Enums.QuestionTypes>(questionDto.QuestionType),
					Marks = questionDto.Marks,
					QuizId = QuizId,
					QuestionChoices = new List<QuestionChoices>()
				};

				await questionRepository.AddAsync(questionEntity);
				await _unitOfWork.SaveChangesAsync(); // Save to get the question ID

				// Add question choices
				if (questionDto.QuestionChoices != null && questionDto.QuestionChoices.Any())
				{
					foreach (var choiceDto in questionDto.QuestionChoices)
					{
						var choiceEntity = new QuestionChoices
						{
							ChoiceText = choiceDto.ChoiceText,
							IsCorrect = choiceDto.IsCorrect,
							QuizQuestionId = questionEntity.Id
						};

						await choiceRepository.AddAsync(choiceEntity);
					}
				}
			}

			await _unitOfWork.SaveChangesAsync();

			// Reload quiz with questions and choices
			var quizSpec = new QuizSpecification(QuizId);
			var updatedQuiz = await quizRepository.GetByIdAsync(quizSpec);
			return updatedQuiz!.Adapt<QuizResponseDto>();
		}


*/
		public async Task DeleteQuizAsync(int quizId)
		{
			var quizRepository = _unitOfWork.GetRepository<Quiz, int>();
			var quizEntity = await quizRepository.GetByIdAsync(quizId);
			if (quizEntity is null)
			{
				throw new QuizNotFoundException(quizId);
			}
			quizRepository.Delete(quizEntity);
			await _unitOfWork.SaveChangesAsync();
		}

		public async Task<IEnumerable<QuizResponseDto>> GetAllQuizzesAsync()
		{
			var quizRepository = _unitOfWork.GetRepository<Quiz, int>();
			var quizSpec = new QuizSpecification();
			var quizEntities = await quizRepository.GetAllAsync(quizSpec);
            return quizEntities.Adapt<IEnumerable<QuizResponseDto>>();
		}

		public async Task<IEnumerable<QuizResponseDto>> GetAllQuizzesForCourse(int CourseId)
		{
			var quizRepository = _unitOfWork.GetRepository<Quiz, int>();
			var quizSpec = new QuizByCourseIdSpecification(CourseId);
			var quizEntities = await quizRepository.GetAllAsync(quizSpec);
			if (quizEntities is null || !quizEntities.Any())
			{
				throw new QuizzesInCourseNotFoundException(CourseId);
            }
            return quizEntities.Adapt<IEnumerable<QuizResponseDto>>();
		}

		public async Task<QuizResponseDto> GetQuizByIdAsync(int quizId)
		{
			var quizRepository = _unitOfWork.GetRepository<Quiz, int>();
			var quizSpec = new QuizSpecification(quizId);
			var quizEntity = await quizRepository.GetByIdAsync(quizSpec);
			if (quizEntity is null)
			{
				throw new QuizNotFoundException(quizId);
			}
			return quizEntity.Adapt<QuizResponseDto>();
		}
	}
}
