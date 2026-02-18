
namespace ServiceAbstractionLayer
{
    public interface ICacheService
    {
        public Task<string?> GetAsync(string key);
        public  Task SetAsync(string key, object value, TimeSpan ttl);
    }

    
}
