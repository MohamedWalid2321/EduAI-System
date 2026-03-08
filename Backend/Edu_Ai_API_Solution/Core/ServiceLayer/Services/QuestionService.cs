using DomainLayer.Exceptions.Question;
using DomainLayer.Models;
using ServiceLayer.Specifications.QuestionSpecifications;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ServiceLayer.Services
{
    public class QuestionService(IUnitOfWork _unitOfWork) : IQuestionService
    {
        public async Task<QuestionResponseDto> CreateQuestionForQuiz(int quizId, QuestionRequestDto questionRequest)
        {
            var quizRepository = _unitOfWork.GetRepository<Quiz, int>();
            var QuestionRepository = _unitOfWork.GetRepository<QuizQuestion, int>();

            // Verify quiz exists
            var quizEntity = await quizRepository.GetByIdAsync(quizId);
            if (quizEntity is null)
            {
                throw new QuizNotFoundException(quizId);
            }

            var questionInQuizSpecifications = new QuestionInQuizSpecifications(quizId, questionRequest.QuestionText);

            //check the duplicate of question in the same quiz
            var count = await QuestionRepository.GetCountAsync(questionInQuizSpecifications);
            if (count>0)
            {
                throw new QuestionAlreadyExistsException(questionRequest.QuestionText);
            }

            //check the choice count
            var choiceCount = questionRequest.QuestionChoices.Count();
            if (choiceCount < 2)
            {
                throw new NotEnoughChoicesException();
            }
            if (choiceCount > 4)
            {
                throw new TooManyChoicesException();
            }
            //check the duplicate of choices
            var duplicateChoices = questionRequest.QuestionChoices.Distinct().Count();
            if (duplicateChoices != choiceCount)
            {
                throw new DuplicateChoicesException();
            }
            //check the correct answer index
            if (questionRequest.CorrectAnswerIndex < 0 || questionRequest.CorrectAnswerIndex >= choiceCount)
            {
                throw new InvalidCorrectAnswerIndexException();
            }

            var questionEntity = questionRequest.Adapt<QuizQuestion>();
            questionEntity.QuizId = quizId;
            await QuestionRepository.AddAsync(questionEntity);
            await _unitOfWork.SaveChangesAsync();
            return questionEntity.Adapt<QuestionResponseDto>();
        }


        public async Task<QuestionResponseDto> ToggleQuestionAsync(int quizId, int questionId)
        {
            var quizRepository = _unitOfWork.GetRepository<Quiz, int>();
            var QuestionRepository = _unitOfWork.GetRepository<QuizQuestion, int>();

            // Verify quiz exists
            var quizEntity = await quizRepository.GetByIdAsync(quizId);
            if (quizEntity is null)
            {
                throw new QuizNotFoundException(quizId);
            }

            // Verify question exists
            var questionSpecifications = new QuestionInQuizSpecifications(quizId, questionId);

            var questionEntity = await QuestionRepository.GetByIdAsync(questionSpecifications);
            if (questionEntity is null)
            {
                throw new QuestionNotFoundException(questionId);
            }


            // Toggle the IsActive status of the question
            questionEntity.IsActive = !questionEntity.IsActive;
            QuestionRepository.Update(questionEntity);
            await _unitOfWork.SaveChangesAsync();
            return questionEntity.Adapt<QuestionResponseDto>();

        }

        public async Task<QuestionResponseDto> UpdateQuestionAsync(int quizId, QuestionRequestDto questionRequest)
        {
            var quizRepository = _unitOfWork.GetRepository<Quiz, int>();
            var QuestionRepository = _unitOfWork.GetRepository<QuizQuestion, int>();

            // Verify quiz exists
            var quizEntity = await quizRepository.GetByIdAsync(quizId);

            if (quizEntity is null)
            {
                throw new QuizNotFoundException(quizId);
            }

            // Verify question exists
            var questionSpecifications = new QuestionInQuizSpecifications(quizId, questionRequest.Id);

            var questionEntity = await QuestionRepository.GetByIdAsync(questionSpecifications);

            if (questionEntity is null)
            {
                throw new QuestionNotFoundException(questionRequest.Id);
            }

            
            var questionInQuizSpecifications = new QuestionInQuizSpecifications(quizId, questionRequest.QuestionText);
            //check the duplicate of question in the same quiz
            var count = await QuestionRepository.GetCountAsync(questionInQuizSpecifications);
            if (count > 0)
            {
                throw new QuestionAlreadyExistsException(questionRequest.QuestionText);
            }

            //check the choice count
            var choiceCount = questionRequest.QuestionChoices.Count();
            if (choiceCount < 2)
            {
                throw new NotEnoughChoicesException();
            }
            if (choiceCount > 4)
            {
                throw new TooManyChoicesException();
            }
            //check the duplicate of choices
            var duplicateChoices = questionRequest.QuestionChoices.Distinct().Count();
            if (duplicateChoices != choiceCount)
            {
                throw new DuplicateChoicesException();
            }
            //check the correct answer index
            if (questionRequest.CorrectAnswerIndex < 0 || questionRequest.CorrectAnswerIndex >= choiceCount)
            {
                throw new InvalidCorrectAnswerIndexException();
            }

            var questionEntityToUpdate = await QuestionRepository.GetByIdAsync(questionSpecifications);
            questionEntityToUpdate.QuestionText = questionRequest.QuestionText;
            questionEntityToUpdate.QuestionType = Enum.Parse<DomainLayer.Enums.QuestionTypes>(questionRequest.QuestionType);
            questionEntityToUpdate.Marks = questionRequest.Marks;

            questionEntity.QuestionChoices.Clear();
            questionEntity.QuestionChoices = questionRequest.QuestionChoices
                .Select((choiceText, index) => new DomainLayer.Models.QuestionChoices
                {
                    ChoiceText = choiceText,
                    IsCorrect = index == questionRequest.CorrectAnswerIndex
                }).ToList();

            QuestionRepository.Update(questionEntityToUpdate);
            await _unitOfWork.SaveChangesAsync();
            return questionEntity.Adapt<QuestionResponseDto>();



        }
    }
}
