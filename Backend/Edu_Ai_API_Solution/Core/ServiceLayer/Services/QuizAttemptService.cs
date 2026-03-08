using DomainLayer.Exceptions.AttemptQuiz;
using DomainLayer.Exceptions.Question;
using DomainLayer.Models;
using Mapster;
using Microsoft.AspNetCore.Identity;
using ServiceLayer.Specifications.AttemptedQuizSpecification;
using Shared.Dtos.AttemptQuiz;
using Shared.Dtos.AttemptQuiz.Request;
using Shared.Dtos.AttemptQuiz.Response;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ServiceLayer.Services
{
    public class QuizAttemptService(IUnitOfWork _unitOfWork) : IQuizAttemptService
    {
        public async Task<StartQuizResponseDto> StartQuizAsync(string quizCode, string studentId)
        {
            var attemptRepository = _unitOfWork.GetRepository<QuizAttempt, int>();
            var quizRepository = _unitOfWork.GetRepository<Quiz, int>();
            

            var quizSpecification = new QuizSpecification(quizCode);
            var quizEntity = await quizRepository.GetFirstOrDefaultAsync(quizSpecification);

            if (quizEntity is null)
            {
                throw new QuizCodeNotFoundException(quizCode);
            }

            var hasAttemptedQuizSpecification = new HasAttemptedQuizSpecification(quizEntity.Id, studentId);
            var hasAttemptedQuiz = await attemptRepository.GetCountAsync(hasAttemptedQuizSpecification) > 0;
            if (hasAttemptedQuiz)
            {
                throw new QuizAlreadyAttemptedException(quizCode, studentId);
            }

            var quizAttemptEntity = new QuizAttempt
            {

                QuizId = quizEntity.Id,
                QuizCode = quizCode,
                StudentId = studentId,
                CreatedAt = DateTime.UtcNow

            };
            
            quizAttemptEntity.Score = 0; // initialize score to 0 when starting the quiz
            quizAttemptEntity.Quiz = quizEntity; // set the navigation property to the quiz entity
            

            await attemptRepository.AddAsync(quizAttemptEntity);
            await _unitOfWork.SaveChangesAsync();

            var response = new StartQuizResponseDto
            {
                AttemptId = quizAttemptEntity.Id,
                Title = quizEntity.Title,
                Duration = quizEntity.Duration,
                Questions = quizEntity.QuizQuestions
                                    .Select(q => new QuestionForStudentDto
                                    {
                                        //Id = q.Id,
                                        QuestionText = q.QuestionText,
                                        Choices = q.QuestionChoices.Select(c => new ChoiceDto
                                        {
                                            //Id = c.Id,
                                            ChoiceText = c.ChoiceText
                                        }).ToList()
                                    }).ToList()
            };
            return response;  
        }

        public async Task<SubmitQuizResponseDto> SubmitQuizAsync(
            int attemptId,
            SubmitQuizRequestDto request,
            string studentId)
        {
            var attemptRepository = _unitOfWork.GetRepository<QuizAttempt, int>();
            var quizRepository = _unitOfWork.GetRepository<Quiz, int>();
            var studentAnswerRepository = _unitOfWork.GetRepository<StudentAnswer, int>();


            var quizAttemptSpecification = new QuizAttemptSpecification(attemptId, studentId);

            var attemptedQuiz = await attemptRepository.GetFirstOrDefaultAsync(quizAttemptSpecification);

            if (attemptedQuiz is null)
                throw new QuizAttemptNotFoundException(studentId);



            var quizSpecification = new QuizSpecification(attemptedQuiz.QuizId);

            var quizEntity = await quizRepository.GetFirstOrDefaultAsync(quizSpecification);

            if (quizEntity is null)
                throw new QuizNotFoundException(attemptedQuiz.QuizId);

            if (attemptedQuiz.IsSubmitted) { 
                throw new QuizAlreadySubmittedException(studentId);
            }

            if (!attemptedQuiz.CreatedAt.HasValue)
                throw new AttemptCreatedAtException();

            var timeElapsed = DateTime.UtcNow - attemptedQuiz.CreatedAt.Value;
            if (timeElapsed > attemptedQuiz.Quiz.Duration)
            {
                throw new QuizTimeExpiredException();
            }

            int score = 0;

            var studentAnswers = new List<StudentAnswer>();


            foreach (var answer in request.Answers)
            {
                
                var question = quizEntity.QuizQuestions
                    .FirstOrDefault(q => q.Id == answer.QuestionId);

                if (question is null)
                    throw new QuestionNotFoundException(answer.QuestionId);

                var correctChoice = question.QuestionChoices
                    .FirstOrDefault(c => c.IsCorrect);

                if (answer.ChoiceId <=0)
                    throw new SelectedChoiceNotFoundException(question.Id);

                bool isCorrect = correctChoice?.Id == answer.ChoiceId;

                if (isCorrect)
                    score++;


                studentAnswers.Add(new StudentAnswer
                {
                    QuizAttemptId = attemptId,
                    QuizQuestionId = answer.QuestionId,
                    QuestionChoiceId = answer.ChoiceId,
                    IsCorrect = isCorrect
                });
            }


            foreach (var studentAnswer in studentAnswers)
            {
                await studentAnswerRepository.AddAsync(studentAnswer);
            }


            attemptedQuiz.Score = score;
            attemptedQuiz.IsSubmitted = true;
            attemptedQuiz.SubmittedAt = DateTime.UtcNow;



            await _unitOfWork.SaveChangesAsync();


            var answersDict = studentAnswers.ToDictionary(a => a.QuizQuestionId);

            var response = new SubmitQuizResponseDto
            {
                QuizCode = attemptedQuiz.QuizCode,
                QuizTitle = quizEntity.Title,
                Score = score,
                TotalQuestions = quizEntity.QuizQuestions.Count,

                Questions = quizEntity.QuizQuestions.Select(q =>
                {
                    answersDict.TryGetValue(q.Id, out var studentAnswer);

                    var studentChoice = q.QuestionChoices
                        .FirstOrDefault(c => c.Id == studentAnswer?.QuestionChoiceId);

                    var correctChoice = q.QuestionChoices
                        .FirstOrDefault(c => c.IsCorrect);

                    return new QuestionResultDto
                    {
                        QuestionText = q.QuestionText,
                        StudentChoice = studentChoice?.ChoiceText,
                        IsCorrect = studentAnswer?.IsCorrect ?? false,
                        CorrectChoice = correctChoice?.ChoiceText
                    };
                }).ToList()
            };

            return response;
        }

        public async Task<SubmitQuizResponseDto> GetQuizResultAsync(int attemptId, string studentId)
        {
            var attemptRepository = _unitOfWork.GetRepository<QuizAttempt, int>();
            var quizRepository = _unitOfWork.GetRepository<Quiz, int>();
            var studentAnswerRepository = _unitOfWork.GetRepository<StudentAnswer, int>();


            var quizAttemptSpecification = new QuizAttemptSpecification(attemptId, studentId);

            var attemptedQuiz = await attemptRepository.GetFirstOrDefaultAsync(quizAttemptSpecification);

            if (attemptedQuiz is null)
                throw new QuizAttemptNotFoundException(studentId);



            var quizSpecification = new QuizSpecification(attemptedQuiz.QuizId);

            var quizEntity = await quizRepository.GetFirstOrDefaultAsync(quizSpecification);

            if (quizEntity is null)
                throw new QuizNotFoundException(attemptedQuiz.QuizId);

            if (!attemptedQuiz.IsSubmitted)
            {
                var timeElapsed = DateTime.UtcNow - attemptedQuiz.CreatedAt;
                if (timeElapsed > attemptedQuiz.Quiz.Duration)
                    throw new QuizTimeExpiredException();
                throw new QuizNotSubmittedException(studentId);
            }

            var existingAnswersSpecification = new ExistingStudentAnswersSpecification(attemptId);

            var studentAnswers = await studentAnswerRepository.GetAllAsync(existingAnswersSpecification);

            var answersDict = studentAnswers.ToDictionary(a => a.QuizQuestionId);

            var response = new SubmitQuizResponseDto
            {
                QuizCode = attemptedQuiz.QuizCode,
                QuizTitle = quizEntity.Title,
                Score = attemptedQuiz.Score,
                TotalQuestions = quizEntity.QuizQuestions.Count,

                Questions = quizEntity.QuizQuestions.Select(q =>
                {
                    answersDict.TryGetValue(q.Id, out var studentAnswer);

                    var studentChoice = q.QuestionChoices
                        .FirstOrDefault(c => c.Id == studentAnswer?.QuestionChoiceId);

                    var correctChoice = q.QuestionChoices
                        .FirstOrDefault(c => c.IsCorrect);

                    return new QuestionResultDto
                    {
                        QuestionText = q.QuestionText,
                        StudentChoice = studentChoice?.ChoiceText,
                        IsCorrect = studentAnswer?.IsCorrect ?? false,
                        CorrectChoice = correctChoice?.ChoiceText
                    };
                }).ToList()
            };

            return response;
        }

        public async Task<List<StudentAttemptDto>> GetStudentsByQuizAsync(string quizCode)
        {
            var attemptRepository = _unitOfWork.GetRepository<QuizAttempt, int>();

            var spec = new QuizAttemptsByQuizSpecification(quizCode);

            var attempts = await attemptRepository.GetAllAsync(spec);

            var result = attempts.Select(a => new StudentAttemptDto
            {
                StudentId = a.StudentId,
                Score = a.Score,
                SubmittedAt = a.SubmittedAt
            }).ToList();

            return result;
        }

        public async Task<List<StudentQuizDto>> GetStudentQuizzesAsync(string studentId)
        {
            var attemptRepository = _unitOfWork.GetRepository<QuizAttempt, int>();

            var spec = new StudentAttemptsSpecification(studentId);

            var attempts = await attemptRepository.GetAllAsync(spec);

            var result = attempts.Select(a => new StudentQuizDto
            {
                QuizTitle = a.Quiz.Title,
                QuizCode = a.Quiz.QuizCode,
                Score = a.Score,
                SubmittedAt = a.SubmittedAt 
            }).ToList();

            return result;
        }
    }
}
