using Microsoft.Extensions.Hosting;
using System.Net.Http;
namespace ServiceLayer
{
	public static class ApplicationServicesRegisteration
	{
		public static IServiceCollection AddApplicationServices(this IServiceCollection services, IHostEnvironment environment)
		{
			// Add application services registrations here
			services.AddHttpClient<IFileStorageService, BunnyNetService>(client =>
			{
				client.Timeout = TimeSpan.FromMinutes(40);
			})
			.ConfigurePrimaryHttpMessageHandler(() =>
			{
				var handler = new HttpClientHandler();
				
				// Only disable revocation check in development
				if (environment.IsDevelopment())
				{
					handler.CheckCertificateRevocationList = false;
				}
				
				return handler;
			});

			services.AddScoped<IServiceManager, ServiceManager>();
			services.AddMapsterConf();

			services.AddSingleton<IRedisService, RedisService>();
			services.AddScoped<ICacheService, CacheService>();
			services.AddHttpContextAccessor();
			services.AddScoped<IAuthunticationService, AuthService>();
			services.AddScoped<IUserService, UserService>();
			services.AddScoped<IRoleService, RoleService>();
			return services;
		}
		
		private static IServiceCollection AddMapsterConf(this IServiceCollection services)
		{
			var mappingConfig = TypeAdapterConfig.GlobalSettings;
			mappingConfig.Scan(Assembly.GetExecutingAssembly());

			services.AddSingleton<IMapper>(new Mapper(mappingConfig));

			return services;
		}
	}
}
