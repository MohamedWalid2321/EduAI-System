namespace persistenceLayer.Email
{
	public class EmailService(IOptions<MailSettings> MailSettings) : IEmailSender
	{
		private readonly MailSettings _mailSettings = MailSettings.Value;
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
			// This email body
			var builder = new BodyBuilder
			{
				HtmlBody = htmlMessage
			};
			message.Body = builder.ToMessageBody();
			// Send the email using SMTP
			using var smtp = new SmtpClient();
			smtp.Connect(_mailSettings.Host, _mailSettings.Port, SecureSocketOptions.StartTls);
			smtp.Authenticate(_mailSettings.Mail, _mailSettings.Password);
			await smtp.SendAsync(message);
			smtp.Disconnect(true);
		}
	}
}
