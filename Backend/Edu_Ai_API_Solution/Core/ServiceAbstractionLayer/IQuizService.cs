<<<<<<< HEAD
﻿using Shared.Dtos.QuizDto;
using Shared.Dtos.QuizDto.Request;
using Shared.Dtos.QuizDto.Response;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ServiceAbstractionLayer
=======
﻿namespace ServiceAbstractionLayer
>>>>>>> f283ebec1b7f11684dfeff6e9246326d74ada2d9
{
	public interface IQuizService
	{
		Task<QuizResponseDto> CreateOrUpdateQuizAync(int CourseId,QuizRequestDto quizRequest);
		Task<QuizResponseDto> GetQuizByIdAsync(int quizId);
		Task<IEnumerable<QuizResponseDto>> GetAllQuizzesAsync();
		Task<IEnumerable<QuizResponseDto>> GetAllQuizzesForCourse(int CourseId);
		Task DeleteQuizAsync(int quizId);
		Task<QuizResponseDto> AddQuestionToQuiz(int QuizId, ICollection<QuizQuestionDto> Questions);
		
	}
}
