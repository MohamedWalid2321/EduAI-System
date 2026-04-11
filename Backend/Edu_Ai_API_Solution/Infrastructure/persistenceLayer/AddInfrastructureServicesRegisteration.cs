using Microsoft.AspNetCore.Identity.UI.Services;
using persistenceLayer.Email;

namespace persistenceLayer
{
	public static class AddInfrastructureServicesRegisteration
	{
		public static IServiceCollection AddInfrastructureServices(this IServiceCollection services,IConfiguration configuration)
		{
			services.AddDbContext<LmsDBContext>(options =>
			{
				options.UseSqlServer(configuration.GetConnectionString("DefaultConnection"), sqlServerOptionsAction: sqlOptions =>
				{
					sqlOptions.UseQuerySplittingBehavior(QuerySplittingBehavior.SplitQuery);
				});
			});
			services.Configure<MailSettings>(configuration.GetSection(nameof(MailSettings)));
			services.AddScoped<IEmailSender, EmailService>();
			services.AddScoped<IEmailBodyBuilder, EmailBodyBuilder>();
			services.AddScoped<IUnitOfWork, UnitOfWork>();
			services.AddAuthConf(configuration);



			return services;
		}
		public static IServiceCollection AddAuthConf(this IServiceCollection services, IConfiguration configuration)
		{

			services.AddScoped<IJwtProvider, JwtProvider>();
			services.Configure<IdentityOptions>(options =>
			{
				options.Password.RequiredLength = 6;
				options.SignIn.RequireConfirmedEmail = true;
				options.User.RequireUniqueEmail = true;
			});
			services.Configure<JwtOptions>(configuration.GetSection(JwtOptions.SectionName));
			// This approach allows for validation of the options using data annotations and ensures that the configuration is valid at startup.
			services.AddOptions<JwtOptions>()
				.BindConfiguration(JwtOptions.SectionName)
				.ValidateDataAnnotations()
				.ValidateOnStart();
			//Retrieve the JWT options from the configuration to use in the authentication setup
		   var jwtSettings = configuration.GetSection(JwtOptions.SectionName).Get<JwtOptions>() ?? throw new InvalidOperationException("JWT options not found in configuration.");

			services.AddAuthentication(options =>
			{
				options.DefaultAuthenticateScheme = JwtBearerDefaults.AuthenticationScheme;
				options.DefaultChallengeScheme = JwtBearerDefaults.AuthenticationScheme;
			}
			).AddJwtBearer(services =>
			{
				services.SaveToken = true;
				services.TokenValidationParameters = new TokenValidationParameters
				{
					ValidateIssuer = true,
					ValidIssuer = jwtSettings.Issuer,
					ValidateAudience = true,
					ValidAudience = jwtSettings.Audience,
					ValidateLifetime = true,
					IssuerSigningKey = new Microsoft.IdentityModel.Tokens.SymmetricSecurityKey(System.Text.Encoding.UTF8.GetBytes(jwtSettings.Key)),
					ValidateIssuerSigningKey = true,
					ClockSkew = TimeSpan.Zero
				};
			});

			return services;
		}

	}
}
