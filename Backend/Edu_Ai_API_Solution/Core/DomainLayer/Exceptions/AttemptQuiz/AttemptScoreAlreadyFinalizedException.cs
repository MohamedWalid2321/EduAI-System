namespace DomainLayer.Exceptions.AttemptQuiz
{
    public sealed class AttemptScoreAlreadyFinalizedException(int attemptId)
        : ConflictException($"The score for attempt with ID {attemptId} has already been finalized and cannot be updated again.")
    {
    }
}
