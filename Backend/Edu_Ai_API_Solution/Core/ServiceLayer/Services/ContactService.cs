using Microsoft.AspNetCore.Identity.UI.Services;
using Microsoft.Extensions.Configuration;
using ServiceAbstractionLayer;
using Shared.Dtos.ContactDto;

namespace ServiceLayer.Services
{
    public class ContactService(
        IEmailSender _emailSender,
        IEmailBodyBuilder _emailBodyBuilder,
        IConfiguration _configuration) : IContactService
    {
        private readonly string _teamEmail =
            _configuration["MailSettings:Mail"]
            ?? throw new InvalidOperationException("MailSettings:Mail is not configured.");

        public async Task SendContactEmailAsync(ContactRequestDto request, CancellationToken cancellationToken = default)
        {
            var templateModel = new Dictionary<string, string>
            {
                { "{{fullName}}",       System.Net.WebUtility.HtmlEncode(request.FullName) },
                { "{{senderEmail}}",    System.Net.WebUtility.HtmlEncode(request.EmailAddress) },
                { "{{subject}}",        System.Net.WebUtility.HtmlEncode(request.Subject) },
                { "{{messageContent}}", System.Net.WebUtility.HtmlEncode(request.MessageContent)
                                            .Replace("\n", "<br/>") }
            };

            var htmlBody = _emailBodyBuilder.GenerateEmailBody("ContactUs", templateModel);

            await _emailSender.SendEmailAsync(
                email: _teamEmail,
                subject: $"[Contact Us] {request.Subject}",
                htmlMessage: htmlBody);
        }
    }
}
