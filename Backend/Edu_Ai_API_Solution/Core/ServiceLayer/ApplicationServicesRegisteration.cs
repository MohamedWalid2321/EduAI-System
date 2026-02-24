namespace ServiceLayer
{
	public static class ApplicationServicesRegisteration
	{
		public static IServiceCollection AddApplicationServices(this IServiceCollection services)
		{
			// Add application services registrations here
			services.AddHttpClient<IFileStorageService, BunnyNetService>();
			services.AddScoped<IServiceManager, ServiceManager>();
			services.AddMapsterConf();

            services.AddSingleton<IRedisService, RedisService>();
            services.AddScoped<ICacheService, CacheService>();
			services.AddHttpContextAccessor();
			services.AddScoped<IAuthunticationService, AuthService>();
			services.AddScoped<IUserService, UserService>();
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
