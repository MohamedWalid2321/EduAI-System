using DomainLayer.Contracts;
using Mapster;
using MapsterMapper;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Http;
using ServiceAbstractionLayer;
using ServiceLayer.Services;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;

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
