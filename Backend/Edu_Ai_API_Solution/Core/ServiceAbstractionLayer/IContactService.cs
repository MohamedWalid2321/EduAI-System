using Shared.Dtos.ContactDto;

namespace ServiceAbstractionLayer
{
    public interface IContactService
    {
        /// <summary>
        /// Sends a contact-us email to the team inbox on behalf of the sender.
        /// </summary>
        Task SendContactEmailAsync(ContactRequestDto request, CancellationToken cancellationToken = default);
    }
}
