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
        public async Task<QuestionResponseDto> CreateQuestionForQuiz(int quizId, QuestionRequestDto questionRequest, CancellationToken cancellationToken = default)
        {
            var quizRepository = _unitOfWork.GetRepository<Quiz, int>();
            var QuestionRepository = _unitOfWork.GetRepository<QuizQuestion, int>();

            // Verify quiz exists
            var quizEntity = await quizRepository.GetByIdAsync(quizId, cancellationToken);
            if (quizEntity is null)
            {
                throw new QuizNotFoundException(quizId);
            }
            var questionInQuizSpecifications = new QuestionInQuizSpecifications(quizId, questionRequest.QuestionText);

            //check the duplicate of question in the same quiz
            var count = await QuestionRepository.GetCountAsync(questionInQuizSpecifications, cancellationToken);
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

            var questionEntity = questionRequest.Adapt<QuizQuestion>();
            questionEntity.QuizId = quizId;
            await QuestionRepository.AddAsync(questionEntity, cancellationToken);
            await _unitOfWork.SaveChangesAsync(cancellationToken);
            return questionEntity.Adapt<QuestionResponseDto>();
        }


        public async Task<QuestionResponseDto> ToggleQuestionAsync(int quizId, int questionId, CancellationToken cancellationToken = default)
        {
            var quizRepository = _unitOfWork.GetRepository<Quiz, int>();
            var QuestionRepository = _unitOfWork.GetRepository<QuizQuestion, int>();

            // Verify quiz exists
            var quizEntity = await quizRepository.GetByIdAsync(quizId, cancellationToken);
            if (quizEntity is null)
            {
                throw new QuizNotFoundException(quizId);
            }

            // Verify question exists
            var questionSpecifications = new QuestionInQuizSpecifications(quizId, questionId);

            var questionEntity = await QuestionRepository.GetByIdAsync(questionSpecifications, cancellationToken);
            if (questionEntity is null)
            {
                throw new QuestionNotFoundException(questionId);
            }


            // Toggle the IsActive status of the question
            questionEntity.IsActive = !questionEntity.IsActive;
            QuestionRepository.Update(questionEntity);
            await _unitOfWork.SaveChangesAsync(cancellationToken);
            return questionEntity.Adapt<QuestionResponseDto>();

        }

        public async Task<QuestionResponseDto> UpdateQuestionAsync(int quizId, QuestionRequestDto questionRequest, CancellationToken cancellationToken = default)
        {
            var quizRepository = _unitOfWork.GetRepository<Quiz, int>();
            var QuestionRepository = _unitOfWork.GetRepository<QuizQuestion, int>();
            var ChoicesRepository = _unitOfWork.GetRepository<QuestionChoices, int>();

            // Verify quiz exists
            var quizEntity = await quizRepository.GetByIdAsync(quizId, cancellationToken);

            if (quizEntity is null)
            {
                throw new QuizNotFoundException(quizId);
            }

            // Verify question exists (and eagerly load its choices)
            var questionSpecifications = new QuestionInQuizSpecifications(quizId, questionRequest.Id);

            var questionEntity = await QuestionRepository.GetByIdAsync(questionSpecifications, cancellationToken);

            if (questionEntity is null)
            {
                throw new QuestionNotFoundException(questionRequest.Id);
            }

            // Bug fix 1: exclude the current question from the duplicate-text check so that
            // keeping the same QuestionText doesn't trigger a false QuestionAlreadyExistsException.
            var questionInQuizSpecifications = new QuestionInQuizSpecifications(quizId, questionRequest.QuestionText, questionRequest.Id);
            var count = await QuestionRepository.GetCountAsync(questionInQuizSpecifications, cancellationToken);
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

            questionEntity.QuestionText = questionRequest.QuestionText;
            questionEntity.QuestionType = (QuestionTypes)questionRequest.QuestionType;
            questionEntity.Marks = questionRequest.Marks;
            questionEntity.IsAllowableToLookDown = questionRequest.IsAllowableToLookDown;

            // Bug fix 2: explicitly remove the old choices from the DB instead of just calling
            // .Clear() on the in-memory collection. Merely clearing the collection tells EF Core
            // to detach the children but does NOT issue DELETE statements, so SaveChanges then
            // INSERTs the new choices on top — producing duplicates.
            foreach (var oldChoice in questionEntity.QuestionChoices.ToList())
            {
                ChoicesRepository.Delete(oldChoice);
            }
            questionEntity.QuestionChoices.Clear();

            questionEntity.QuestionChoices = questionRequest.QuestionChoices
                .Select((choiceText, index) => new DomainLayer.Models.QuestionChoices
                {
                    ChoiceText = choiceText,
                    IsCorrect = index == questionRequest.CorrectAnswerIndex
                }).ToList();

            QuestionRepository.Update(questionEntity);
            await _unitOfWork.SaveChangesAsync(cancellationToken);
            return questionEntity.Adapt<QuestionResponseDto>();
        }
    }
}
