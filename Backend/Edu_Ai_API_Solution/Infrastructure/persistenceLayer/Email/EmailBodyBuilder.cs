namespace persistenceLayer.Email
{
	public class EmailBodyBuilder:IEmailBodyBuilder
	{
		public  string GenerateEmailBody(string template, Dictionary<string, string> TemplateModel)
		{
			var assembly = Assembly.GetExecutingAssembly();
			var resourceName = $"persistenceLayer.Email.Templates.{template}.html";
			
			using var stream = assembly.GetManifestResourceStream(resourceName)
				?? throw new FileNotFoundException($"Template '{template}' not found");
			
			using var reader = new StreamReader(stream);
			var emailBody = reader.ReadToEnd();
			
			foreach (var item in TemplateModel)
			{
				emailBody = emailBody.Replace(item.Key, item.Value);
			}
			
			return emailBody;
		}
	}
}
