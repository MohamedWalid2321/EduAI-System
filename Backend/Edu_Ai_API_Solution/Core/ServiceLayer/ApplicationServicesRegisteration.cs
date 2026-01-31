using Microsoft.Extensions.DependencyInjection;
using ServiceAbstractionLayer;
using ServiceLayer.Services;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.Extensions.Http;

namespace ServiceLayer
{
	public static class ApplicationServicesRegisteration
	{
		public static IServiceCollection AddApplicationServices(this IServiceCollection services)
		{
			// Add application services registrations here
			services.AddHttpClient<IFileStorageService, BunnyNetService>();
			return services;
		}
	}
}
