using Hangfire;
using Microsoft.Extensions.Hosting;
using ServiceLayer.Jobs;
using System.Net.Http;

namespace ServiceLayer
{
	public static class ApplicationServicesRegisteration
	{
		public static IServiceCollection AddApplicationServices(this IServiceCollection services, IHostEnvironment environment, IConfiguration configuration)
		{
			services.AddHttpClient<IFileStorageService, BunnyNetService>(client =>
			{
				client.Timeout = TimeSpan.FromMinutes(40);
			})
			.ConfigurePrimaryHttpMessageHandler(() =>
			{
				var handler = new HttpClientHandler();

				if (environment.IsDevelopment())
				{
					handler.CheckCertificateRevocationList = false;
					handler.ServerCertificateCustomValidationCallback = (message, cert, chain, errors) => true;
				}
				return handler;
			});

			services.AddScoped<IServiceManager, ServiceManager>();
			services.AddMapsterConf()
				.AddBackgroundJobsConfig(configuration);

			services.AddSingleton<IRedisService, RedisService>();
			services.AddScoped<ICacheService, CacheService>();
			services.AddHttpContextAccessor();
			services.AddScoped<IAuthunticationService, AuthService>();
			services.AddScoped<IUserService, UserService>();
			services.AddScoped<IRoleService, RoleService>();
			services.AddScoped<INotificationService, NotificationService>();
			services.AddScoped<AssignmentDeadlineReminderJob>();
			services.AddScoped<WelcomeNotificationJob>();           // ← add this
			services.AddHttpClient<PaymobService>();
			services.AddScoped<IPaymobService, PaymobService>();
			services.AddScoped<IEnrollmentService, EnrollmentService>();
			services.AddScoped<ILectureService, LectureService>();
			services.AddScoped<ICheatingReportService, CheatingReportService>();

			// Risk Analysis
			services.AddScoped<IRiskAnalysisService, RiskAnalysisService>();
			services.AddScoped<RiskScoreCalculationJob>();

			// Contact Us
			services.AddScoped<IContactService, Services.ContactService>();

			return services;
		}

		private static IServiceCollection AddMapsterConf(this IServiceCollection services)
		{
			var mappingConfig = TypeAdapterConfig.GlobalSettings;
			mappingConfig.Scan(Assembly.GetExecutingAssembly());
			services.AddSingleton<IMapper>(new Mapper(mappingConfig));
			return services;
		}

		private static IServiceCollection AddBackgroundJobsConfig(this IServiceCollection services,
		IConfiguration configuration)
		{
			services.AddHangfire(config => config
				.SetDataCompatibilityLevel(CompatibilityLevel.Version_180)
				.UseSimpleAssemblyNameTypeSerializer()
				.UseRecommendedSerializerSettings()
				.UseSqlServerStorage(configuration.GetConnectionString("DefaultConnection")));

			services.AddHangfireServer();
			return services;
		}
	}
}
