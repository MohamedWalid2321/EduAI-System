using MailKit.Net.Smtp;
using MailKit.Security;
using Microsoft.AspNetCore.Identity.UI.Services;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Options;
using MimeKit;

namespace persistenceLayer.Email
{
	public class EmailService(IOptions<MailSettings> mailSettings, IHostEnvironment environment) : IEmailSender
	{
		private readonly MailSettings _mailSettings = mailSettings.Value;
		private readonly IHostEnvironment _environment = environment;

		public async Task SendEmailAsync(string email, string subject, string htmlMessage)
		{
			// Define the email message
			var message = new MimeMessage
			{
				Sender = MailboxAddress.Parse(_mailSettings.Mail),
				Subject = subject
			};
			// this is for multiple email addresses
			message.To.Add(MailboxAddress.Parse(email));
			
			var builder = new BodyBuilder
			{
				HtmlBody = htmlMessage
			};
			message.Body = builder.ToMessageBody();
			// Send the email using SMTP
			using var smtp = new SmtpClient();
			
			// Only bypass SSL validation in development
			if (_environment.IsDevelopment())
			{
				smtp.CheckCertificateRevocation = false;
				smtp.ServerCertificateValidationCallback = (s, c, h, e) => true;
			}
			
			await smtp.ConnectAsync(_mailSettings.Host, _mailSettings.Port, SecureSocketOptions.StartTls);
			await smtp.AuthenticateAsync(_mailSettings.Mail, _mailSettings.Password);
			await smtp.SendAsync(message);
			await smtp.DisconnectAsync(true);
		}
	}
}
