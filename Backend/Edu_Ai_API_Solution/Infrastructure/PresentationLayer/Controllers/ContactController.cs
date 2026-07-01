using Shared.Dtos.ContactDto;

namespace PresentationLayer.Controllers
{
    /// <summary>
    /// Handles "Contact Us" messages from users/visitors.
    /// The submitted message is forwarded as a formatted email to the Lumina team inbox.
    /// </summary>
    public class ContactController(IServiceManager serviceManager) : ApiControllerBase
    {
        private readonly IServiceManager _serviceManager = serviceManager;

        /// <summary>
        /// Sends a contact-us email to the Lumina team.
        /// </summary>
        /// <remarks>
        /// No authentication required — anyone can reach out to the team.
        ///
        /// **Request Body**
        /// | Field          | Type   | Required | Max Length | Notes                     |
        /// |----------------|--------|----------|------------|---------------------------|
        /// | fullName       | string | ✅       | 150        | Sender's display name     |
        /// | emailAddress   | string | ✅       | —          | Sender's reply-to address |
        /// | subject        | string | ✅       | 250        | Email subject             |
        /// | messageContent | string | ✅       | 5000       | Body of the message       |
        ///
        /// **Responses**
        /// - `200 OK` — Email sent successfully.
        /// - `400 Bad Request` — Validation failed (missing / malformed fields).
        /// - `500 Internal Server Error` — SMTP failure or server error.
        /// </remarks>
        [HttpPost]
        [AllowAnonymous]
        [ProducesResponseType(StatusCodes.Status200OK)]
        [ProducesResponseType(StatusCodes.Status400BadRequest)]
        public async Task<IActionResult> SendContactEmail(
            [FromBody] ContactRequestDto request,
            CancellationToken cancellationToken)
        {
            await _serviceManager.ContactService.SendContactEmailAsync(request, cancellationToken);

            return Ok(new
            {
                Message = "Your message has been sent successfully. We'll get back to you as soon as possible."
            });
        }
    }
}
